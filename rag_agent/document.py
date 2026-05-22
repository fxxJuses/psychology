import os
import sys
import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import List

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from . import logger

import fitz
from pdf2image import convert_from_path
import pytesseract


@contextmanager
def _suppress_stderr():
    """Temporarily suppress stderr to silence MuPDF warnings."""
    old_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


@dataclass
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 80
    strategy: str = "sentence"   # "sentence" | "recursive"
    overlap_sentences: int = 2


OCR_DPI = 200
OCR_LANG = "chi_sim+eng"
OCR_PAGE_BATCH = 20


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, respecting both Chinese and English boundaries."""
    text = text.strip()
    if not text:
        return []

    sentences = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in "。！？…~!?\n":
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())

    merged = []
    accum = ""
    for s in sentences:
        if len(accum + s) < 30:
            accum += s
        else:
            if accum:
                merged.append(accum)
            accum = s
    if accum:
        merged.append(accum)

    return merged


def _sentence_chunk(
    document: Document, config: ChunkingConfig
) -> List[Document]:
    """Sentence-aware chunking: never break mid-sentence, carry overlap as context."""
    text = document.page_content
    source = document.metadata.get("source", "")
    sentences = _split_sentences(text)
    if not sentences:
        return []

    target = config.chunk_size
    overlap_count = config.overlap_sentences

    chunks: List[Document] = []
    i = 0

    while i < len(sentences):
        buf_sentences: List[str] = []
        buf_len = 0
        while i < len(sentences):
            s = sentences[i]
            if buf_len + len(s) > target and buf_len > 0:
                break
            buf_sentences.append(s)
            buf_len += len(s)
            i += 1

        if not buf_sentences:
            buf_sentences = [sentences[i]]
            buf_len = len(sentences[i])
            i += 1

        content = "\n".join(buf_sentences).strip()
        if len(content) < 20:
            continue

        chunks.append(Document(
            page_content=content,
            metadata={
                "source": source,
                "chunk_strategy": "sentence",
            },
        ))

        # carry overlap
        if i < len(sentences):
            overlap_start = max(0, i - overlap_count)
            i = overlap_start

    return chunks


def _recursive_chunk(
    document: Document, config: ChunkingConfig
) -> List[Document]:
    """Legacy recursive character split."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "：", "，",
                     ". ", "? ", "! ", " ", ""],
    )
    return splitter.split_documents([document])


def _pdf_has_text(filepath: str) -> bool:
    with _suppress_stderr():
        doc = fitz.open(filepath)
        try:
            for i in range(min(10, doc.page_count)):
                text = doc[i].get_text().strip()
                if len(text) > 20:
                    return True
            return False
        finally:
            doc.close()


def _load_pdf_with_fitz(filepath: str) -> List[Document]:
    """Load PDF using fitz directly, suppressing MuPDF warnings."""
    docs: List[Document] = []
    filename = os.path.basename(filepath)

    with _suppress_stderr():
        doc = fitz.open(filepath)
        try:
            total = doc.page_count
            skipped = 0
            for i in range(total):
                text = doc[i].get_text().strip()
                if text and len(text) > 10:
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": filepath, "page": i + 1},
                    ))
                else:
                    skipped += 1
        finally:
            doc.close()

    print(f"[Document] 加载 {filename}: {len(docs)} 页"
          f"{f' (跳过 {skipped} 个空白页)' if skipped else ''}")
    return docs


def _load_pdf_with_ocr(filepath: str) -> List[Document]:
    doc = fitz.open(filepath)
    total_pages = doc.page_count
    doc.close()

    logger.info(f"[OCR] 开始识别扫描版 PDF: {os.path.basename(filepath)}, "
                f"共 {total_pages} 页, DPI={OCR_DPI}, lang={OCR_LANG}")
    docs: List[Document] = []
    accumulated = ""
    page_start = 1

    for batch_start in range(0, total_pages, OCR_PAGE_BATCH):
        batch_end = min(batch_start + OCR_PAGE_BATCH, total_pages)
        logger.info(f"[OCR] {os.path.basename(filepath)}: "
                    f"处理第 {batch_start + 1}-{batch_end} 页 / 共 {total_pages}")

        pages = convert_from_path(
            filepath,
            dpi=OCR_DPI,
            first_page=batch_start + 1,
            last_page=batch_end,
        )

        for i, image in enumerate(pages):
            page_num = batch_start + i + 1
            text = pytesseract.image_to_string(image, lang=OCR_LANG).strip()
            if text:
                accumulated += f"\n--- 第{page_num}页 ---\n{text}"

            if len(accumulated) > 5000 or page_num == total_pages:
                docs.append(Document(
                    page_content=accumulated.strip(),
                    metadata={
                        "source": filepath,
                        "page": page_start,
                        "end_page": page_num,
                        "type": "ocr",
                    },
                ))
                accumulated = ""
                page_start = page_num + 1

        # Release image memory
        for img in pages:
            img.close()
        del pages

    logger.info(f"[OCR] 完成: {os.path.basename(filepath)}, "
                f"产出 {len(docs)} 个文本块")
    return docs


def load_documents(source_dir: str) -> List[Document]:
    logger.info(f"[Document] 加载目录: {source_dir}")

    docs: List[Document] = []

    txt_loader = DirectoryLoader(
        source_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
        show_progress=logger.is_verbose(),
    )
    txt_docs = txt_loader.load()
    if txt_docs:
        docs.extend(txt_docs)
        print(f"[Document] 加载 .txt 文件 {len(txt_docs)} 个")

    pdf_files = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))

    for pdf_path in pdf_files:
        if _pdf_has_text(pdf_path):
            pdf_docs = _load_pdf_with_fitz(pdf_path)
            docs.extend(pdf_docs)
        else:
            ocr_docs = _load_pdf_with_ocr(pdf_path)
            docs.extend(ocr_docs)
            print(f"[Document] OCR 识别 {os.path.basename(pdf_path)}: "
                  f"{len(ocr_docs)} 个文本块")

    return docs


def chunk_documents(
    documents: List[Document], config: ChunkingConfig | None = None
) -> List[Document]:
    cfg = config or ChunkingConfig()
    logger.info(f"[Document] 文档切分: strategy={cfg.strategy}, "
                f"chunk_size={cfg.chunk_size}, overlap={cfg.chunk_overlap}")

    if cfg.strategy == "sentence":
        all_chunks: List[Document] = []
        for doc in documents:
            all_chunks.extend(_sentence_chunk(doc, cfg))
        result = all_chunks
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "：", "，",
                        ". ", "? ", "! ", " ", ""],
        )
        result = splitter.split_documents(documents)

    logger.info(f"[Document] 切分结果: {len(documents)} 文档 -> {len(result)} chunks")
    if logger.is_verbose():
        for i, chunk in enumerate(result[:10]):
            src = chunk.metadata.get("source", "unknown")
            preview = chunk.page_content[:100].replace("\n", " ")
            logger.keyval(f"chunk[{i}]", f"({src}) len={len(chunk.page_content)} {preview}...", indent=6)
        if len(result) > 10:
            logger.keyval("...", f"还有 {len(result) - 10} 个 chunks 未显示")

    return result


CHUNKS_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chunks")


def _safe_stem(source_name: str) -> str:
    """Return a filesystem-safe stem from a source filename."""
    # remove extension, replace unsafe chars
    stem = os.path.splitext(source_name)[0]
    # replace characters that are problematic in filenames
    for ch in r'\/:*?"<>|':
        stem = stem.replace(ch, "_")
    return stem


def save_chunks_to_json(
    chunks: List[Document],
    output_dir: str | None = None,
    config: ChunkingConfig | None = None,
) -> str:
    """Save chunks to per-source JSON files for inspection and tuning.

    Each PDF gets its own JSON file under output_dir. A _manifest.json
    summarises all sources with stats.

    Returns the output directory path.
    """
    out_dir = output_dir or CHUNKS_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # group chunks by source basename
    groups: dict[str, List[dict]] = {}
    stats: dict[str, dict] = {}

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        source_name = os.path.basename(source)
        content = chunk.page_content

        rec = {
            "chunk_id": i,
            "page": chunk.metadata.get("page"),
            "chunk_strategy": chunk.metadata.get("chunk_strategy", ""),
            "char_count": len(content),
            "content": content,
        }

        groups.setdefault(source_name, []).append(rec)

        if source_name not in stats:
            stats[source_name] = {"chunk_count": 0, "total_chars": 0, "min_chars": None, "max_chars": 0}
        s = stats[source_name]
        s["chunk_count"] += 1
        s["total_chars"] += len(content)
        if s["min_chars"] is None or len(content) < s["min_chars"]:
            s["min_chars"] = len(content)
        if len(content) > s["max_chars"]:
            s["max_chars"] = len(content)

    # write per-source JSON files
    saved_files: list[str] = []
    for source_name, records in groups.items():
        stem = _safe_stem(source_name)
        filename = f"{stem}_{timestamp}.json"
        filepath = os.path.join(out_dir, filename)

        payload = {
            "source": source_name,
            "generated_at": timestamp,
            "chunk_size": config.chunk_size if config else None,
            "chunk_overlap": config.chunk_overlap if config else None,
            "chunk_strategy": config.strategy if config else None,
            "chunk_count": len(records),
            "chunks": records,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        saved_files.append(filepath)

    # compute averages for stats
    for s in stats.values():
        s["avg_chars"] = round(s["total_chars"] / s["chunk_count"], 1)
        del s["total_chars"]

    # write manifest
    manifest = {
        "generated_at": timestamp,
        "total_chunks": len(chunks),
        "chunk_size": config.chunk_size if config else None,
        "chunk_overlap": config.chunk_overlap if config else None,
        "chunk_strategy": config.strategy if config else None,
        "per_source_stats": stats,
        "files": [os.path.basename(f) for f in saved_files],
    }
    manifest_path = os.path.join(out_dir, "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"[Document] 切分结果已保存到: {out_dir}")
    print(f"[Document] 切分结果已保存到: {out_dir}")
    for fp in saved_files:
        print(f"  - {os.path.basename(fp)}")
    print(f"  - _manifest.json")
    return out_dir
