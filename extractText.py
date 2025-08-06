
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
    found_ack = False
    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        lines = page_text.splitlines()

        # 1ページ目のタイトル・著者名（最初の3-5行程度）を除外
        if page_num == 0:
            # タイトル・著者名・所属などを除去（最初の5行をスキップ）
            lines = lines[5:] if len(lines) > 7 else lines[2:]

        # ヘッダー・フッター（最初と最後の1-2行）を除外
        lines = lines[1:-1] if len(lines) > 2 else lines

        # ページ番号（数字のみの行）や空行を除去
        lines = [line for line in lines if not re.match(r'^\s*\d+\s*$', line)]
        lines = [line for line in lines if line.strip() != ""]

        # "Acknowledgments"や「謝辞」以降は無視
        if any(re.match(r'^(Acknowledg?ments?|謝辞)', line, re.IGNORECASE) for line in lines):
            idx = next(i for i, line in enumerate(lines) if re.match(r'^(Acknowledg?ments?|謝辞)', line, re.IGNORECASE))
            lines = lines[:idx]
            found_ack = True

        # "References"や「参考文献」以降は無視
        if any(re.match(r'^(References|参考文献)', line, re.IGNORECASE) for line in lines):
            idx = next(i for i, line in enumerate(lines) if re.match(r'^(References|参考文献)', line, re.IGNORECASE))
            lines = lines[:idx]

        # 1行だけの大文字タイトルや短いノイズ行を除去（例: SECTION, ABSTRACT, KEYWORDS など）
        lines = [line for line in lines if not re.match(r'^[A-Z\- ]{3,}$', line.strip())]
        lines = [line for line in lines if len(line.strip()) > 3]

        text += "\n".join(lines) + "\n"
        if found_ack:
            break  # 謝辞以降は無視
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