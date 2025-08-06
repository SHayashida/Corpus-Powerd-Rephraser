
# Corpus-Powered Rephraser

**バージョン: 1.1.0**

## 概要
学術論文PDFからコーパスを自動抽出・ベクトル化し、ユーザーのドラフト文に対して意味的に近い表現を検索し、OpenAI APIを用いて高品質なリライト案を生成するOSSツールです。

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
1. 論文PDFを `pdfs/` フォルダにまとめて保存（`pdfs/` フォルダは自動検出されます）
2. OpenAI APIキーを環境変数に設定
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```
3. `extractText.py` を実行してコーパス抽出・ベクトル化・FAISSインデックス作成
   ```bash
   python extractText.py
   ```
4. `rephrase.py` でドラフト文の意味検索・リライト案生成
   - 対話形式で入力
   - コマンドライン引数で直接指定
   - 自動英訳＋英語論文コーパスからリフレーズも可能

### 便利な使い方例

- **日本語ドラフト文を自動で英訳し、英語論文コーパスからリフレーズ案を得る**
  ```bash
  python rephrase.py --sentence "本研究では新しい手法を提案する。" --auto-translate
  ```
  → 英訳＋英語論文表現で3案を自動生成

- **英語ドラフト文を直接リフレーズ**
  ```bash
  python rephrase.py --sentence "We propose a new method for this problem."
  ```

- **Jupyterや他Pythonスクリプトから関数として呼び出し**
  ```python
  from rephrase import search_similar_sentences, generate_rephrase
  similar = search_similar_sentences("your draft", model, index, all_sentences)
  print(generate_rephrase("your draft", similar))
  ```

- **MCPや外部AIツールからAPI的に利用**
  - `rephrase.py` の関数をimportし、ドラフト文を自動で英訳→リフレーズ→AI執筆支援に組み込むことが可能
  - 例: 論文執筆AIの自動英訳・リフレーズ補助

---


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
