# -*- coding: utf-8 -*-
"""离线验证不需要额外依赖的 RAG demo。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "01_document_loading.py",
    "02_chunking.py",
    "03_embeddings_hash.py",
    "06_vectorstore_memory.py",
    "12_retrieval_scores.py",
    "09_rag_naive.py --offline",
    "10_rag_citation.py --offline",
]


def main():
    root = Path(__file__).resolve().parent
    for script in SCRIPTS:
        parts = script.split()
        path = root / parts[0]
        args = parts[1:]
        print(f"\n===== {parts[0]} {' '.join(args)} =====")
        result = subprocess.run([sys.executable, str(path), *args], cwd=root)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
