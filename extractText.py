
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import glob
import os

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        lines = page_text.splitlines()

        # 1ページ目のタイトル（最初の2行）を除外
        if page_num == 0:
            lines = lines[2:]

        # ヘッダー・フッター（最初と最後の1-2行）を除外
        lines = lines[1:-1] if len(lines) > 2 else lines

        # ページ番号（数字のみの行）を除去
        lines = [line for line in lines if not re.match(r'^\s*\d+\s*$', line)]

        # "References"以降は無視
        if any(re.match(r'^(References|参考文献)', line, re.IGNORECASE) for line in lines):
            idx = next(i for i, line in enumerate(lines) if re.match(r'^(References|参考文献)', line, re.IGNORECASE))
            lines = lines[:idx]

        text += "\n".join(lines) + "\n"
    return text

# 日本語も考慮した高性能な多言語モデルをロード
model = SentenceTransformer('all-mpnet-base-v2')

def process_text_to_vectors(pdf_path):
    # テキストを抽出
    corpus_text = extract_text_from_pdf(pdf_path)

    # 文単位に分割（より洗練された分割方法も検討可能）
    sentences = re.split(r'(?<=[。！？.!?])\s+', corpus_text)

    # 各文をベクトルに変換（エンベディング）
    print("Embedding corpus sentences...")
    corpus_embeddings = model.encode(sentences, convert_to_tensor=True)
    print(f"Successfully created {len(corpus_embeddings)} vectors.")

    return sentences, corpus_embeddings

def build_faiss_index(corpus_embeddings):
    # FAISSインデックスを作成
    d = corpus_embeddings.shape[1]  # ベクトルの次元数
    index = faiss.IndexFlatL2(d)
    index.add(corpus_embeddings.cpu().numpy())  # FAISSはnumpy配列を要求
    print(f"FAISS index created with {index.ntotal} vectors.")
    return index


# --- PDFフォルダ内の全PDFファイルを自動で取得 ---
PDF_DIR = "pdfs"  # まとめて保存するフォルダ名
pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))

all_sentences = []
all_embeddings = []
for pdf_path in pdf_files:
    sentences, corpus_embeddings = process_text_to_vectors(pdf_path)
    all_sentences.extend(sentences)
    all_embeddings.append(corpus_embeddings)
import torch
if all_embeddings:
    all_embeddings = torch.cat(all_embeddings, dim=0)
    index = build_faiss_index(all_embeddings)
else:
    all_embeddings = None
    index = None

if __name__ == "__main__":
    print(all_sentences[:5])  # 最初の5文を表示
    print(all_embeddings.shape)  # ベクトルの形状を表示
    print(f"FAISS index created with {index.ntotal} vectors.")