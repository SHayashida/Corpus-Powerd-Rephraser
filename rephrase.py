
import torch
import os
import argparse
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
from extractText import process_text_to_vectors, build_faiss_index

# --- 既存のコーパスベクトル・文リストのロード例（必要に応じてファイル保存/ロード処理を追加可能） ---
# ここでは extractText.py で作成した all_sentences, all_embeddings, index を前提とします。
# 実際の運用では pickle等で保存・ロードすることも推奨

# --- ステップ2: ドラフト文のベクトル化と意味検索 ---
def search_similar_sentences(user_sentence, model, index, sentences, k=3):
    # ユーザー文をベクトル化
    query_vector = model.encode([user_sentence])
    # FAISSで類似文検索
    distances, indices = index.search(query_vector, k)
    # 検索結果の文を取得
    similar_sentences = [sentences[i] for i in indices[0]]
    return similar_sentences

# --- ステップ2: LLMによるリライト生成 ---
def generate_rephrase(original_sentence, similar_sentences, lang_hint=None):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if lang_hint == "en":
        prompt = f"""
You are an expert academic editor. Rewrite the following sentence in three different, sophisticated, and natural academic English styles, referencing the style and vocabulary of the provided top paper examples. Do not copy-paste, but keep the core meaning and key entities/claims.

---
# Original sentence:
{original_sentence}

# Reference sentences (from top papers):
- {similar_sentences[0].strip()}
- {similar_sentences[1].strip()}
- {similar_sentences[2].strip()}
---

# Suggestions:
1.
"""
    else:
        prompt = f"""
あなたは学術論文の専門エディターです。
以下の「元の文」を、分野の専門家が書いたような、より洗練された自然な学術的表現に書き換えてください。

制約条件:
- 「元の文」の核心的な意味（キーエンティティ、主張）は絶対に維持してください。
- 以下の「参考文」のスタイルや語彙を参考にしてください。ただし、単なるコピー＆ペーストは避けてください。
- 3つの異なる表現を提案してください。

---
# 元の文:
{original_sentence}

# 参考文 (この分野のトップ論文の表現例):
- {similar_sentences[0].strip()}
- {similar_sentences[1].strip()}
- {similar_sentences[2].strip()}
---

# 提案:
1. 
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
        n=1
    )
    return response.choices[0].message.content

# --- 自動英訳機能 ---
def translate_to_english(text):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt = f"""
Translate the following academic sentence to natural, sophisticated English suitable for a scientific paper. Only output the translation.
---
{text}
---
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=120,
        n=1
    )
    return response.choices[0].message.content.strip()

# --- テスト例 ---

if __name__ == "__main__":
    # extractText.py で作成したモデル・インデックス・文リストをimportする例
    from extractText import model, index, all_sentences

    parser = argparse.ArgumentParser(description="Corpus-powered rephraser with auto-translation option.")
    parser.add_argument("--sentence", type=str, help="Draft sentence to rephrase.")
    parser.add_argument("--auto-translate", action="store_true", help="If set, auto-translate Japanese input to English and rephrase using English corpus.")
    args = parser.parse_args()

    if args.sentence:
        user_draft_sentence = args.sentence.strip()
    else:
        user_draft_sentence = input("Enter your draft sentence: ").strip()

    if args.auto_translate:
        # 日本語→英語自動翻訳
        print("Translating to English...")
        en_sentence = translate_to_english(user_draft_sentence)
        print(f"[English translation] {en_sentence}")
        similar_sentences = search_similar_sentences(en_sentence, model, index, all_sentences, k=3)
        rephrased_suggestions = generate_rephrase(en_sentence, similar_sentences, lang_hint="en")
    else:
        similar_sentences = search_similar_sentences(user_draft_sentence, model, index, all_sentences, k=3)
        rephrased_suggestions = generate_rephrase(user_draft_sentence, similar_sentences)

    print("\n--- AI-Generated Rephrasing Suggestions ---")
    print(rephrased_suggestions)

all_sentences = []
all_embeddings = []

pdf_files = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
for pdf_path in pdf_files:
    sentences, corpus_embeddings = process_text_to_vectors(pdf_path)
    all_sentences.extend(sentences)
    all_embeddings.append(corpus_embeddings)

import torch
all_embeddings = torch.cat(all_embeddings, dim=0)
index = build_faiss_index(all_embeddings)
