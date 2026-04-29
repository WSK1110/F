#!/opt/anaconda3/bin/python3
"""
批量导入 10k_files/ 下的 PDF 到 Chroma，跳过 web upload。
支持增强解析（pdfplumber 表格提取）和配置化路径。
"""
import sys
import os
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RAG_core import RAGChatbotCore
from config import DATA_DIR, CHROMA_DIR


def main():
    pdf_files = sorted(Path(DATA_DIR).glob('*.pdf'))
    if not pdf_files:
        print("No PDF found in", DATA_DIR)
        return

    print(f"Found {len(pdf_files)} PDFs:")
    for f in pdf_files:
        print(f"  - {f.name}")

    r = RAGChatbotCore()
    r.config['EMBEDDINGS']['provider'] = 'ollama'
    r.config['EMBEDDINGS']['model'] = 'mxbai-embed-large'

    print("\n[1/3] Loading PDFs (enhanced parser)...")
    documents = r.load_pdfs([str(f) for f in pdf_files], use_enhanced_parser=True)

    tables = sum(1 for d in documents if d.metadata.get('type') == 'table')
    print(f"  → {len(documents)} segments loaded ({tables} tables detected)")

    print("\n[2/3] Creating vector store...")
    vector_store = r.create_vector_store(documents)
    if not vector_store:
        print("  → ERROR: vector store creation failed")
        return
    print(f"  → Chroma collection: {CHROMA_DIR} ({len(r.document_splits)} chunks)")

    print("\n[3/3] Creating QA chain...")
    qa_chain = r.create_qa_chain()
    if not qa_chain:
        print("  → ERROR: QA chain creation failed")
        return
    print("  → QA chain ready")

    print(f"\n✅ All 10-K PDFs ingested successfully.")
    print(f"   Collection: {CHROMA_DIR}")
    print(f"   Total chunks: {len(r.document_splits)}")
    print(f"   Tables extracted: {tables}")


if __name__ == '__main__':
    main()
