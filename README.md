# HuggingFace LLM Chatbot with RAG - Electron + FastAPI

ローカルで動作するLLMチャットボットアプリケーションです。HuggingFaceのモデルを使用し、RAG（Retrieval-Augmented Generation）機能により、アップロードしたドキュメントを参照した回答が可能です。

## 特徴

- 🤖 複数のLLMモデルを選択可能
- 💬 リアルタイムチャットインターフェース
- 📚 **RAG機能**: PDF/TXT/MDファイルをアップロードして参照可能
- 🔍 **ベクトル検索**: FAISS + HuggingFace Embeddingsによる高速検索
- 🎛️ Temperature、最大長などのパラメータ調整
- 🔄 モデルの動的ロード・切り替え
- 💾 モデルキャッシュによる高速化

## プロジェクト構成

```
.
├── backend/              # FastAPIサーバー
│   ├── Pipfile          # Python依存関係
│   ├── main.py          # FastAPI + HuggingFace + RAG実装
│   ├── rag_manager.py   # RAG機能（ドキュメント処理・検索）
│   ├── uploads/         # アップロードファイル保存先
│   └── vector_store/    # FAISSベクトルDB永続化
├── frontend/            # Electronアプリ
│   ├── package.json     # Node.js依存関係
│   ├── main.js          # Electronメインプロセス
│   └── index.html       # チャットボットUI（RAG対応）
└── README.md
```

## 必要な環境

- Python 3.12+
- Node.js 18.0.0+
- pipenv
- (オプション) CUDA対応GPU（高速化のため）

## セットアップ

### バックエンド（FastAPI + HuggingFace）

1. backendディレクトリに移動：
```bash
cd backend
```

2. pipenvをインストール（まだの場合）：
```bash
pip install pipenv
```

3. 依存関係をインストール：
```bash
pipenv install
```

**注意**: 初回インストール時、PyTorch、Transformersなどの大きなパッケージをダウンロードするため、時間がかかります。

### フロントエンド（Electron）

1. frontendディレクトリに移動：
```bash
cd frontend
```

2. 依存関係をインストール：
```bash
npm install
```

## 実行方法

### 1. FastAPIサーバーを起動

backendディレクトリで：
```bash
cd backend
pipenv run python main.py
```

サーバーは http://localhost:8000 で起動します。

### 2. Electronアプリを起動

別のターミナルを開き、frontendディレクトリで：
```bash
cd frontend
npm start
```

## 使い方

### 基本的な使い方

1. Electronアプリが起動したら、上部のモデル選択ドロップダウンからモデルを選択
2. 「読み込み」ボタンをクリックしてモデルをロード（初回は数分かかる場合があります）
3. メッセージ入力欄にテキストを入力して送信
4. LLMからの応答を待つ

### RAG機能の使い方

1. **ドキュメントをアップロード**
   - 左サイドバーの「📁 ファイルをアップロード」エリアをクリック
   - PDF、TXT、またはMDファイルを選択
   - アップロード完了後、ドキュメント一覧に表示されます

2. **RAGを使った質問**
   - チャット設定で「RAGを使用」をチェック
   - 「取得数」でドキュメントから検索する情報の件数を設定（1-10件）
   - 質問を入力して送信
   - アップロードしたドキュメントから関連情報を検索し、回答に反映されます

3. **ドキュメント管理**
   - サイドバーでアップロード済みドキュメントを確認
   - 「全て削除」ボタンですべてのドキュメントを削除可能

## 利用可能なモデル

### 軽量モデル（推奨）
- **gpt2**: OpenAIのGPT-2（英語、最も軽量）
- **rinna-japanese-gpt2**: 日本語GPT-2モデル（軽量）

### 中〜大サイズモデル
- **gpt2-medium**: GPT-2 Medium（英語）
- **rinna-japanese-gpt-1b**: 日本語GPT 1Bモデル
- **cyberagent-calm2**: CyberAgent CALM2 7B（日本語、大きめ、高性能だがメモリ要件大）

**メモリ要件**:
- 軽量モデル: 2-4GB RAM
- 中サイズ: 8-16GB RAM
- 大サイズ: 16GB+ RAM推奨

## パラメータ設定

- **最大長**: 生成するテキストの最大トークン数（50-500）
- **Temperature**: 生成のランダム性（0.1-2.0）
  - 低い値（0.1-0.5）: より決定的で一貫性のある出力
  - 高い値（1.0-2.0）: よりクリエイティブで多様な出力

## APIエンドポイント

FastAPIサーバーは以下のエンドポイントを提供します：

### LLMモデル関連
- `GET /` - API情報
- `GET /models` - 利用可能なモデル一覧
- `GET /model/current` - 現在ロードされているモデル情報
- `POST /model/select` - モデルを選択してロード
- `POST /chat` - チャットメッセージを送信（RAG対応）
- `GET /health` - ヘルスチェック

### RAG機能関連
- `POST /documents/upload` - ドキュメントをアップロード（PDF/TXT/MD）
- `GET /documents/list` - アップロード済みドキュメント一覧
- `DELETE /documents/delete` - 全ドキュメントを削除

### その他
- `GET /docs` - Swagger UI（APIドキュメント）

### APIの使用例

```bash
# モデル一覧を取得
curl http://localhost:8000/models

# モデルをロード
curl -X POST http://localhost:8000/model/select \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gpt2"}'

# 通常のチャット
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you?",
    "model_id": "gpt2",
    "max_length": 100,
    "temperature": 0.7,
    "use_rag": false
  }'

# RAGを使ったチャット
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "アップロードしたファイルの内容を教えて",
    "model_id": "gpt2",
    "use_rag": true,
    "rag_k": 3
  }'

# ドキュメントをアップロード
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@document.pdf"

# ドキュメント一覧を取得
curl http://localhost:8000/documents/list
```

## トラブルシューティング

### モデルのロードが遅い
- 初回ロード時はHuggingFace Hubからモデルをダウンロードするため時間がかかります
- ダウンロード後はキャッシュされるため、2回目以降は高速です
- キャッシュ場所: `~/.cache/huggingface/`

### メモリ不足エラー
- より軽量なモデル（gpt2など）を試してください
- 他のアプリケーションを閉じてメモリを確保してください

### CUDA/GPUエラー
- CPUモードで動作しますが、GPUがある場合は自動的に使用されます
- CUDA関連のエラーが出る場合、CPUモードで問題なく動作します

### サーバーに接続できない
- FastAPIサーバーが起動しているか確認してください
- ポート8000が他のアプリケーションで使用されていないか確認してください

## 開発

### FastAPIサーバー
- [main.py](backend/main.py) を編集
- 変更後、サーバーを再起動（`pipenv run python main.py`）

### Electronアプリ
- [index.html](frontend/index.html) でUIを編集
- [main.js](frontend/main.js) でElectronの設定を編集
- 変更を確認するにはアプリを再起動（`npm start`）

### 新しいモデルを追加
[main.py](backend/main.py:25-46) の `AVAILABLE_MODELS` 辞書に新しいモデルを追加：

```python
AVAILABLE_MODELS = {
    "your-model-id": {
        "name": "huggingface/model-name",
        "description": "モデルの説明"
    }
}
```

## ライセンス

MIT

## 注意事項

- このアプリケーションはローカルで動作し、データは外部に送信されません
- 大きなモデルを使用する場合、十分なメモリとストレージ容量が必要です
- モデルの使用には各モデルのライセンス条項が適用されます
