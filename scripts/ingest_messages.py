import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb


def _extract_pairs(payload: Any) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []

    if isinstance(payload, dict):
        q = payload.get("question") or payload.get("user") or payload.get("prompt")
        a = payload.get("answer") or payload.get("assistant") or payload.get("response")
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
            pairs.append((q.strip(), a.strip()))

        for value in payload.values():
            pairs.extend(_extract_pairs(value))
        return pairs

    if isinstance(payload, list):
        for item in payload:
            pairs.extend(_extract_pairs(item))
        return pairs

    return pairs


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ingest(messages_dir: Path, chroma_dir: Path, collection_name: str) -> int:
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(name=collection_name)

    total = 0
    for json_file in messages_dir.rglob("*.json"):
        try:
            payload = _read_json(json_file)
            pairs = _extract_pairs(payload)
            if not pairs:
                continue

            documents: List[str] = []
            metadatas: List[Dict[str, str]] = []
            ids: List[str] = []

            for question, answer in pairs:
                documents.append(f"Question: {question}\nAnswer: {answer}")
                metadatas.append({"source": str(json_file), "question": question, "answer": answer})
                ids.append(str(uuid.uuid4()))

            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            total += len(pairs)
        except Exception as exc:
            print(f"Skipping {json_file}: {exc}")

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest message JSON files into ChromaDB.")
    parser.add_argument("--messages-dir", required=True, help="Folder containing JSON messages.")
    parser.add_argument("--chroma-dir", default="data/chroma", help="ChromaDB persistence path.")
    parser.add_argument("--collection", default="imex_messages", help="Collection name.")
    args = parser.parse_args()

    total = ingest(Path(args.messages_dir), Path(args.chroma_dir), args.collection)
    print(f"Ingestion complete. Added {total} Q&A pairs.")


if __name__ == "__main__":
    main()
