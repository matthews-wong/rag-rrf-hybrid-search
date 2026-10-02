import argparse
import sys

from .search import HybridSearcher, load_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(prog="hybrid_search")
    parser.add_argument("query")
    parser.add_argument("docs", help="JSONL file of {id, text} objects")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    docs = {r["id"]: r["text"] for r in load_jsonl(args.docs)}
    for doc_id, score in HybridSearcher(docs).search(args.query, k=args.k):
        print(f"{score:.4f}\t{doc_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
