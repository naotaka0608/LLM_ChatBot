# ソフトウェア仕様書 - HuggingFace LLM Chatbot

**バージョン**: 1.0
**作成日**: 2025-11-09
**プロジェクト名**: HuggingFace LLM Chatbot (Electron + FastAPI)

---

## 1. 概要

### 1.1 システム概要
ローカル環境で動作するLLMチャットボットアプリケーション。HuggingFaceの事前学習済みモデルを使用し、プライバシーを保護しながらAIチャットを実現する。

### 1.2 技術スタック
- **バックエンド**: Python 3.12 + FastAPI + HuggingFace Transformers
- **フロントエンド**: Node.js 24.1.0 + Electron
- **パッケージ管理**: pipenv (Python), npm (Node.js)
- **機械学習**: PyTorch, Transformers

### 1.3 システム構成
```
クライアント (Electron)
    ↓ HTTP (localhost:8000)
サーバー (FastAPI)
    ↓
HuggingFace Transformers
    ↓
ローカルLLMモデル
```

---

## 2. プロジェクト構造

```
project-root/
├── backend/
│   ├── Pipfile              # Python依存関係定義
│   ├── Pipfile.lock         # 自動生成（pipenv install後）
│   └── main.py              # FastAPIサーバー実装
├── frontend/
│   ├── package.json         # Node.js依存関係定義
│   ├── package-lock.json    # 自動生成（npm install後）
│   ├── main.js              # Electronメインプロセス
│   ├── index.html           # チャットボットUI
│   └── node_modules/        # 自動生成（npm install後）
├── README.md                # ユーザー向けドキュメント
└── SoftwareSpecifications.md # 本仕様書
```

---

## 3. バックエンド仕様（FastAPI）

### 3.1 依存関係（Pipfile）

```toml
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
fastapi = "*"
uvicorn = {extras = ["standard"], version = "*"}
transformers = "*"
torch = "*"
accelerate = "*"
sentencepiece = "*"
protobuf = "*"

[dev-packages]

[requires]
python_version = "3.12"
```

### 3.2 APIエンドポイント仕様

#### 3.2.1 ルート - `GET /`
**説明**: API情報とエンドポイント一覧を返す

**レスポンス例**:
```json
{
  "message": "HuggingFace LLM Chatbot API",
  "endpoints": {
    "/models": "利用可能なモデル一覧",
    "/chat": "チャット（POST）",
    "/model/select": "モデル選択（POST）",
    "/model/current": "現在のモデル情報"
  }
}
```

#### 3.2.2 モデル一覧取得 - `GET /models`
**説明**: 利用可能なモデル一覧と現在ロードされているモデルを返す

**レスポンス例**:
```json
{
  "models": {
    "rinna-japanese-gpt2": {
      "name": "rinna/japanese-gpt2-medium",
      "description": "日本語GPT-2モデル（軽量）"
    },
    "gpt2": {
      "name": "gpt2",
      "description": "GPT-2（英語、軽量）"
    }
  },
  "current_model": "gpt2"
}
```

#### 3.2.3 現在のモデル情報 - `GET /model/current`
**説明**: 現在ロードされているモデルの情報を返す

**レスポンス例**:
```json
{
  "model_id": "gpt2",
  "model_info": {
    "name": "gpt2",
    "description": "GPT-2（英語、軽量）"
  }
}
```

#### 3.2.4 モデル選択 - `POST /model/select`
**説明**: 指定されたモデルをロード（またはキャッシュから取得）

**リクエストボディ**:
```json
{
  "model_id": "gpt2"
}
```

**レスポンス例**:
```json
{
  "status": "success",
  "model_id": "gpt2",
  "model_info": {
    "name": "gpt2",
    "description": "GPT-2（英語、軽量）"
  }
}
```

**エラーレスポンス例**:
```json
{
  "detail": "指定されたモデルIDが見つかりません"
}
```

#### 3.2.5 チャット - `POST /chat`
**説明**: メッセージを送信してLLMからの応答を取得

**リクエストボディ**:
```json
{
  "message": "Hello, how are you?",
  "model_id": "gpt2",
  "max_length": 100,
  "temperature": 0.7
}
```

**パラメータ**:
- `message` (required): 入力テキスト
- `model_id` (optional): モデルID（デフォルト: "gpt2"）
- `max_length` (optional): 最大生成長（デフォルト: 100）
- `temperature` (optional): 生成のランダム性（デフォルト: 0.7）

**レスポンス例**:
```json
{
  "status": "success",
  "response": "Hello, how are you? I'm doing great, thanks for asking!",
  "model_id": "gpt2",
  "input": "Hello, how are you?"
}
```

#### 3.2.6 ヘルスチェック - `GET /health`
**説明**: サーバーの状態を確認

**レスポンス例**:
```json
{
  "status": "healthy",
  "cuda_available": false,
  "loaded_models": ["gpt2"],
  "current_model": "gpt2"
}
```

### 3.3 利用可能なモデル定義

```python
AVAILABLE_MODELS = {
    "rinna-japanese-gpt2": {
        "name": "rinna/japanese-gpt2-medium",
        "description": "日本語GPT-2モデル（軽量）"
    },
    "gpt2": {
        "name": "gpt2",
        "description": "GPT-2（英語、軽量）"
    },
    "gpt2-medium": {
        "name": "gpt2-medium",
        "description": "GPT-2 Medium（英語）"
    },
    "rinna-japanese-gpt-1b": {
        "name": "rinna/japanese-gpt-1b",
        "description": "日本語GPT 1Bモデル"
    },
    "cyberagent-calm2": {
        "name": "cyberagent/calm2-7b-chat",
        "description": "CyberAgent CALM2 7B Chat（日本語、大きめ）"
    }
}
```

### 3.4 主要機能実装要件

#### 3.4.1 モデルキャッシュシステム
- ロードしたモデルは`model_cache`辞書に保存
- 同じモデルを再度リクエストされた場合、キャッシュから取得
- キャッシュキー: モデルID
- キャッシュ内容: `{"tokenizer": tokenizer, "model": model, "pipeline": pipe}`

#### 3.4.2 GPU/CPU自動切り替え
- `torch.cuda.is_available()`でGPU利用可能性を判定
- GPU利用可能: `torch.float16`, `device_map="auto"`, `device=0`
- GPU利用不可: `torch.float32`, `device_map=None`, `device=-1`

#### 3.4.3 CORS設定
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### 3.4.4 ロギング
- レベル: INFO
- 出力内容:
  - モデルのロード開始/完了
  - テキスト生成開始
  - エラー情報

### 3.5 起動方法
```bash
cd backend
pipenv run python main.py
```

起動後、`http://0.0.0.0:8000`でリッスン

---

## 4. フロントエンド仕様（Electron）

### 4.1 依存関係（package.json）

```json
{
  "name": "electron-hello-world",
  "version": "1.0.0",
  "description": "Electron Hello World Application",
  "main": "main.js",
  "scripts": {
    "start": "electron ."
  },
  "keywords": [],
  "author": "",
  "license": "MIT",
  "engines": {
    "node": ">=18.0.0"
  },
  "devDependencies": {
    "electron": "^28.0.0"
  }
}
```

### 4.2 Electronメインプロセス（main.js）

#### 4.2.1 ウィンドウ設定
```javascript
{
  width: 800,
  height: 600,
  webPreferences: {
    nodeIntegration: true,
    contextIsolation: false
  }
}
```

#### 4.2.2 ロードファイル
`index.html`をロード

#### 4.2.3 ライフサイクル
- `app.whenReady()`: ウィンドウ作成
- `app.on('activate')`: macOS再アクティベート時にウィンドウ再作成
- `app.on('window-all-closed')`: macOS以外ではアプリ終了

### 4.3 UI仕様（index.html）

#### 4.3.1 レイアウト構造
```
┌─────────────────────────────────────┐
│ Header                              │
│  - タイトル: "🤖 LLM Chatbot"       │
│  - モデル選択ドロップダウン          │
│  - 読み込みボタン                   │
│  - ステータス表示                   │
├─────────────────────────────────────┤
│ Chat Container                      │
│  ┌───────────────────────────────┐  │
│  │ Settings Bar                  │  │
│  │  - 最大長 (number input)      │  │
│  │  - Temperature (range slider) │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ Messages Area                 │  │
│  │  - システムメッセージ          │  │
│  │  - ユーザーメッセージ（右寄せ）│  │
│  │  - ボットメッセージ（左寄せ）  │  │
│  │  (スクロール可能)             │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ Loading Indicator (非表示)    │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ Input Container               │  │
│  │  - テキスト入力フィールド      │  │
│  │  - 送信ボタン                 │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

#### 4.3.2 カラースキーム
- **背景グラデーション**: `#667eea` → `#764ba2`
- **ヘッダー背景**: `rgba(255, 255, 255, 0.95)`
- **ユーザーメッセージ**: `#667eea`（白テキスト）
- **ボットメッセージ**: `#f0f0f0`（黒テキスト）
- **システムメッセージ**: `#fff3cd`（`#856404`テキスト）
- **プライマリボタン**: `#667eea` (hover: `#764ba2`)

#### 4.3.3 UI要素仕様

**モデル選択ドロップダウン**:
- ID: `modelSelect`
- 初期値: "読み込み中..."
- 動的生成: `/models` APIからモデル一覧を取得して選択肢を生成

**読み込みボタン**:
- テキスト: "読み込み"
- クリック時: `loadSelectedModel()` 実行

**ステータス表示**:
- ID: `modelStatus`
- ロード済み時: "✓ ロード済み"（緑色）
- ロード中: "ロード中..."

**最大長入力**:
- ID: `maxLength`
- type: `number`
- デフォルト値: 100
- 範囲: 50-500
- ステップ: 10

**Temperature スライダー**:
- ID: `temperature`
- type: `range`
- デフォルト値: 0.7
- 範囲: 0.1-2.0
- ステップ: 0.1
- 現在値表示: `tempValue`

**メッセージ入力**:
- ID: `messageInput`
- placeholder: "メッセージを入力..."
- Enterキー: 送信実行

**送信ボタン**:
- ID: `sendButton`
- テキスト: "送信"
- 処理中は無効化（disabled）

**メッセージエリア**:
- ID: `messages`
- 自動スクロール: 最下部
- メッセージタイプ:
  - `message user`: 右寄せ、青背景
  - `message bot`: 左寄せ、グレー背景
  - `message system`: 中央、黄色背景

**ローディングインジケーター**:
- ID: `loading`
- デフォルト: 非表示
- 処理中: 表示（クラス: `active`）
- スピナーアニメーション付き

#### 4.3.4 JavaScript関数仕様

**グローバル変数**:
```javascript
const API_URL = 'http://localhost:8000';
let currentModel = null;
let availableModels = {};
```

**主要関数**:

1. `init()`: 初期化（モデル一覧取得、イベントリスナー設定）
2. `loadAvailableModels()`: `/models` APIからモデル一覧を取得してドロップダウンに設定
3. `loadSelectedModel()`: `/model/select` APIでモデルをロード
4. `updateModelStatus()`: モデルステータス表示を更新
5. `updateTemperatureDisplay()`: Temperatureスライダーの値表示を更新
6. `sendMessage()`: `/chat` APIにメッセージを送信
7. `addMessage(type, content)`: メッセージをUIに追加（type: 'user' | 'bot'）
8. `addSystemMessage(content)`: システムメッセージをUIに追加
9. `handleKeyPress(event)`: Enterキーで送信実行

**アニメーション**:
- メッセージ追加時: `fadeIn`（0.3秒、透明度+上方向移動）
- スピナー: 回転アニメーション（1秒）

### 4.4 起動方法
```bash
cd frontend
npm start
```

---

## 5. データフロー

### 5.1 初期化フロー
```
1. Electronアプリ起動
2. index.html読み込み
3. init()実行
4. GET /models → モデル一覧取得
5. ドロップダウンに選択肢を設定
6. 待機状態
```

### 5.2 モデルロードフロー
```
1. ユーザーがモデル選択
2. 「読み込み」ボタンクリック
3. loadSelectedModel()実行
4. POST /model/select → モデルロード開始
5. サーバー側:
   - model_cacheを確認
   - 未キャッシュの場合:
     - HuggingFace Hubからダウンロード
     - Tokenizer、Modelをロード
     - Pipelineを作成
     - キャッシュに保存
6. 成功レスポンス
7. UI更新（ステータス: "✓ ロード済み"）
8. システムメッセージ追加
```

### 5.3 チャットフロー
```
1. ユーザーがメッセージ入力
2. 送信ボタンクリックまたはEnterキー
3. sendMessage()実行
4. ユーザーメッセージをUIに追加
5. 送信ボタン無効化、ローディング表示
6. POST /chat → テキスト生成リクエスト
7. サーバー側:
   - モデルキャッシュから取得
   - pipeline()でテキスト生成
8. 生成されたテキストをレスポンス
9. ボットメッセージをUIに追加
10. ローディング非表示、送信ボタン有効化
```

---

## 6. 環境要件

### 6.1 ソフトウェア要件
- **Python**: 3.12以上
- **Node.js**: 18.0.0以上（推奨: 24.1.0）
- **pipenv**: 最新版
- **npm**: Node.jsに付属

### 6.2 ハードウェア要件

**最小要件**:
- CPU: 2コア以上
- RAM: 4GB以上
- ストレージ: 10GB以上の空き容量

**推奨要件**:
- CPU: 4コア以上
- RAM: 16GB以上
- GPU: CUDA対応GPU（オプション、高速化のため）
- ストレージ: 20GB以上の空き容量

**モデル別メモリ要件**:
- 軽量モデル（gpt2, rinna-japanese-gpt2）: 2-4GB RAM
- 中サイズ（gpt2-medium, rinna-japanese-gpt-1b）: 8-16GB RAM
- 大サイズ（cyberagent-calm2）: 16GB以上 RAM

---

## 7. セットアップ手順

### 7.1 バックエンドセットアップ
```bash
cd backend
pip install pipenv
pipenv install
```

### 7.2 フロントエンドセットアップ
```bash
cd frontend
npm install
```

---

## 8. 起動手順

### 8.1 バックエンド起動
```bash
cd backend
pipenv run python main.py
```
→ `http://0.0.0.0:8000`でリッスン開始

### 8.2 フロントエンド起動（別ターミナル）
```bash
cd frontend
npm start
```
→ Electronウィンドウが開く

---

## 9. 使用方法

1. Electronアプリを起動
2. 上部のドロップダウンからモデルを選択
3. 「読み込み」ボタンをクリック（初回は数分かかる）
4. ステータスが「✓ ロード済み」になるのを確認
5. 最大長とTemperatureを必要に応じて調整
6. メッセージ入力欄にテキストを入力
7. 送信ボタンをクリックまたはEnterキー
8. LLMからの応答を待つ

---

## 10. 機能要件

### 10.1 必須機能
- ✅ 複数LLMモデルの選択機能
- ✅ モデルの動的ロード・切り替え
- ✅ チャットインターフェース
- ✅ リアルタイムメッセージ表示
- ✅ パラメータ調整（最大長、Temperature）
- ✅ モデルキャッシュシステム
- ✅ GPU/CPU自動切り替え
- ✅ エラーハンドリング
- ✅ ローディング表示

### 10.2 非機能要件
- ✅ ローカル実行（インターネット接続不要、モデルダウンロード後）
- ✅ プライバシー保護（データ外部送信なし）
- ✅ レスポンシブUI
- ✅ ログ出力
- ✅ CORS対応

---

## 11. エラーハンドリング

### 11.1 バックエンドエラー
- モデルID不正: HTTP 400, `"指定されたモデルIDが見つかりません"`
- モデルロード失敗: HTTP 500, `"モデルのロードに失敗しました: {エラー詳細}"`
- チャット処理失敗: HTTP 500, `"チャット処理に失敗しました: {エラー詳細}"`

### 11.2 フロントエンドエラー
- サーバー接続失敗: システムメッセージ表示 `"エラー: サーバーに接続できません..."`
- モデル未ロード時の送信: システムメッセージ表示 `"先にモデルをロードしてください。"`
- API通信エラー: システムメッセージ表示 `"エラー: {エラー詳細}"`

---

## 12. セキュリティ要件

### 12.1 ネットワークセキュリティ
- ローカルホストのみ通信（外部公開不可）
- CORS: 全オリジン許可（ローカル環境のため）

### 12.2 データセキュリティ
- ユーザーデータは外部送信なし
- モデルキャッシュはメモリ内のみ（永続化なし）

---

## 13. パフォーマンス要件

### 13.1 応答時間
- モデルロード（初回）: 数十秒〜数分（モデルサイズ依存）
- モデルロード（キャッシュ済み）: 即座
- チャット応答: 数秒〜数十秒（生成長、モデルサイズ、ハードウェア依存）

### 13.2 最適化
- モデルキャッシュによる再ロード高速化
- GPU利用による生成速度向上
- `low_cpu_mem_usage=True`によるメモリ効率化

---

## 14. テストシナリオ

### 14.1 基本動作テスト
1. バックエンド起動確認
2. フロントエンド起動確認
3. モデル一覧取得確認
4. モデルロード成功確認
5. チャット送受信確認

### 14.2 エラーケーステスト
1. サーバー未起動時のフロントエンド動作
2. 不正なモデルID指定
3. モデル未ロード時の送信
4. 空メッセージの送信

### 14.3 パフォーマンステスト
1. 複数モデルの連続ロード
2. キャッシュされたモデルの再ロード速度
3. 長文生成の応答時間

---

## 15. 制限事項

### 15.1 技術的制限
- メモリに収まるサイズのモデルのみ使用可能
- 生成速度はハードウェア性能に依存
- モデルの精度はHuggingFaceの事前学習済みモデルに依存

### 15.2 機能的制限
- 会話履歴の管理なし（単発の生成のみ）
- モデルのファインチューニング機能なし
- 複数ユーザーの同時使用不可

---

## 16. 今後の拡張案

### 16.1 機能拡張
- 会話履歴の保存・読み込み
- マルチターン会話のコンテキスト管理
- モデルのカスタマイズ・ファインチューニング
- ストリーミング応答
- 音声入出力

### 16.2 UI/UX改善
- ダークモード
- カスタムテーマ
- マークダウンレンダリング
- コードハイライト
- 設定の永続化

---

## 17. ライセンス

### 17.1 プロジェクトライセンス
MIT License

### 17.2 使用モデルのライセンス
各HuggingFaceモデルのライセンスに従う:
- GPT-2: Modified MIT License
- Rinna models: MIT License
- CyberAgent CALM2: Apache 2.0

---

## 18. 参考資料

### 18.1 公式ドキュメント
- FastAPI: https://fastapi.tiangolo.com/
- HuggingFace Transformers: https://huggingface.co/docs/transformers/
- Electron: https://www.electronjs.org/docs/
- PyTorch: https://pytorch.org/docs/

### 18.2 モデル情報
- GPT-2: https://huggingface.co/gpt2
- Rinna Japanese GPT-2: https://huggingface.co/rinna/japanese-gpt2-medium
- Rinna Japanese GPT-1B: https://huggingface.co/rinna/japanese-gpt-1b
- CyberAgent CALM2: https://huggingface.co/cyberagent/calm2-7b-chat

---

## 19. バージョン履歴

| バージョン | 日付 | 変更内容 |
|----------|------|---------|
| 1.0 | 2025-11-09 | 初版リリース |

---

**文書終了**
