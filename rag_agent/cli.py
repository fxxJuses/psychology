import argparse
import sys

from dotenv import load_dotenv

from .pipeline import RAGPipeline, RAGConfig, DB_DIR_DEFAULT, DOCS_DIR_DEFAULT
from . import logger as log

load_dotenv()


def _add_verbose_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")


def _add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-k", type=int, default=8, help="候选检索数量 (默认8)")
    parser.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default="hybrid",
                        help="检索模式: vector(语义) / bm25(关键词) / hybrid(混合)")
    parser.add_argument("--no-rerank", action="store_true", help="禁用 LLM 精排")
    parser.add_argument("--rerank-top-k", type=int, default=4, help="精排后保留数量 (默认4)")
    parser.add_argument("--no-rewrite", action="store_true", help="禁用查询改写")


def cmd_ingest(args):
    log.setup(verbose=args.verbose)
    config = RAGConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        collection_name=args.collection,
    )
    pipeline = RAGPipeline(db_dir=args.db_dir, config=config)
    count = pipeline.ingest(docs_dir=args.docs_dir)
    if count == 0:
        sys.exit(1)


def cmd_query(args):
    log.setup(verbose=args.verbose)
    config = RAGConfig(
        top_k=args.top_k,
        collection_name=args.collection,
        retrieval_mode=args.mode,
        enable_rerank=not args.no_rerank,
        rerank_top_k=args.rerank_top_k,
        enable_query_rewrite=not args.no_rewrite,
    )
    pipeline = RAGPipeline(db_dir=args.db_dir, config=config)
    result = pipeline.query(args.question)
    print(f"问题: {result['question']}")
    print(f"回答: {result['answer']}")
    print(f"来源: {', '.join(result['sources'])}")


def cmd_interactive(args):
    log.setup(verbose=args.verbose)
    config = RAGConfig(
        top_k=args.top_k,
        collection_name=args.collection,
        retrieval_mode=args.mode,
        enable_rerank=not args.no_rerank,
        rerank_top_k=args.rerank_top_k,
        enable_query_rewrite=not args.no_rewrite,
    )
    pipeline = RAGPipeline(db_dir=args.db_dir, config=config)
    print("RAG 交互问答 (输入 'exit' 退出)")
    print(f"检索模式: {args.mode} | Rerank: {not args.no_rerank} | 查询改写: {not args.no_rewrite}")
    print("-" * 50)
    while True:
        try:
            q = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not q:
            continue
        if q.lower() == "exit":
            break
        result = pipeline.query(q)
        print(f"\n{result['answer']}")
        if args.show_sources:
            print(f"\n来源: {', '.join(result['sources'])}")


def main():
    parser = argparse.ArgumentParser(description="RAG Agent - 基于检索增强生成的问答系统")
    sub = parser.add_subparsers(dest="command", help="子命令")

    # ingest
    p_ingest = sub.add_parser("ingest", help="导入文档到向量数据库")
    p_ingest.add_argument("--docs-dir", default=DOCS_DIR_DEFAULT, help="文档目录")
    p_ingest.add_argument("--db-dir", default=DB_DIR_DEFAULT, help="向量数据库目录")
    p_ingest.add_argument("--chunk-size", type=int, default=256)
    p_ingest.add_argument("--chunk-overlap", type=int, default=25)
    p_ingest.add_argument("--collection", default="rag_agent")
    _add_verbose_arg(p_ingest)

    # query
    p_query = sub.add_parser("query", help="单次问答")
    p_query.add_argument("question", help="问题")
    p_query.add_argument("--db-dir", default=DB_DIR_DEFAULT)
    p_query.add_argument("--collection", default="rag_agent")
    _add_verbose_arg(p_query)
    _add_retrieval_args(p_query)

    # interactive
    p_chat = sub.add_parser("chat", help="交互式问答")
    p_chat.add_argument("--db-dir", default=DB_DIR_DEFAULT)
    p_chat.add_argument("--collection", default="rag_agent")
    p_chat.add_argument("--show-sources", action="store_true")
    _add_verbose_arg(p_chat)
    _add_retrieval_args(p_chat)

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "chat":
        cmd_interactive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
