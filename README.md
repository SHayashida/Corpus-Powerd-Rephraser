# Corpus-Powered Rephraser

## 概要
学術論文PDFからコーパスを自動抽出・ベクトル化し、ユーザーのドラフト文に対して意味的に近い表現を検索し、OpenAI API（GPT-4o）を用いて高品質なリライト案を生成するOSSツールです。

- PDFテキスト抽出: PyMuPDF
- ベクトル化: Sentence-Transformers
- 類似検索: FAISS
- リライト生成: OpenAI API (GPT-4o)

## 特徴
- ローカルでコーパス前処理・ベクトル化・検索が完結
- リライト部分のみ最先端AIサービスを活用
- 日本語・英語など多言語対応
- Jupyter/VSCode/CLIで利用可能

## インストール
1. Python 3.8以降（推奨: 3.13）
2. 仮想環境（venv）を作成
3. 必要なライブラリをインストール

```bash
pip install openai "sentence-transformers>=2.2.0" faiss-cpu pymupdf numpy
```

## 使い方
1. `paper1.pdf`, `paper2.pdf`, `paper3.pdf` などのPDFを `Corpus-Powerd Rephraser` フォルダに保存
2. OpenAI APIキーを環境変数に設定
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```
3. `extractText.py` でコーパス抽出・ベクトル化・FAISSインデックス作成
4. `rephrase.py` でドラフト文の意味検索・リライト案生成

## サンプル実行
```bash
python extractText.py
python rephrase.py
```

## セキュリティ
- `.gitignore` によりAPIキーやキャッシュ・PDF等はGitHubに含まれません
- APIキーは環境変数で管理してください

## ライセンス
MIT License

## 貢献
Pull Request・Issue歓迎

## 著作権・免責
- 本ツールは学術・研究用途を主目的としています
- OpenAI API利用には別途契約・課金が必要です
