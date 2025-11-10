# データベース実装ドキュメント

## 概要

このドキュメントは、LLM ChatbotアプリケーションにSQLite + FAISSのハイブリッドデータベース構成を実装した内容を説明します。

## アーキテクチャ

### ハイブリッド構成

```
データストレージ:
├── SQLite (chatbot.db)          # 構造化データ
│   ├── chat_history             # チャット履歴
│   ├── documents                # ドキュメントメタデータ
│   └── rag_searches             # RAG検索履歴
│
└── FAISS (vector_store/)        # ベクトルデータ
    ├── index.faiss              # ベクトルインデックス
    └── index.pkl                # メタデータ
```

### 役割分担

- **SQLite**: 構造化データ、履歴、統計情報
- **FAISS**: ベクトル検索（高速類似度検索）

## 実装ファイル

### 1. db_manager.py

SQLiteデータベースを管理するクラス。

**主な機能**:
- テーブル作成・管理
- CRUD操作（作成・読取・更新・削除）
- 統計情報の集計

**テーブル構造**:

#### chat_history テーブル
```sql
CREATE TABLE chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_id TEXT,
    use_rag BOOLEAN DEFAULT 0,
    retrieved_docs_count INTEGER DEFAULT 0
)
```

#### documents テーブル
```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    chunks_count INTEGER NOT NULL,
    characters_count INTEGER NOT NULL,
    upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    file_type TEXT
)
```

#### rag_searches テーブル
```sql
CREATE TABLE rag_searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    retrieved_docs TEXT,  -- JSON形式
    k INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    results_count INTEGER DEFAULT 0
)
```

**主要メソッド**:
- `save_chat()`: チャット履歴を保存
- `get_chat_history()`: チャット履歴を取得
- `save_document()`: ドキュメント情報を保存
- `get_documents()`: ドキュメント一覧を取得
- `save_rag_search()`: RAG検索履歴を保存
- `get_statistics()`: 統計情報を取得

### 2. rag_manager.py (更新)

RAG機能にデータベース連携を追加。

**変更点**:
- コンストラクタに`db_manager`パラメータを追加
- `process_document()`: ドキュメント処理時にDBに保存
- `search()`: 検索時にRAG検索履歴をDBに保存
- `delete_all_documents()`: 削除時にDBからも削除
- `_load_documents_from_db()`: アプリ起動時にDBからドキュメント情報を復元

### 3. main.py (更新)

FastAPI エンドポイントにDB機能を統合。

**追加機能**:
- `DBManager`インスタンスを作成し、`RAGManager`に渡す
- チャット時にセッションIDを生成・管理
- チャット履歴をDBに自動保存
- 新しいエンドポイントを追加:
  - `/chat/history/{session_id}`: セッション別履歴
  - `/chat/history`: 全履歴
  - `/rag/searches`: RAG検索履歴
  - `/rag/popular-queries`: 人気クエリ
  - `/statistics`: 統計情報

**セッションID管理**:
```python
# ChatRequestにsession_idフィールドを追加
session_id = request.session_id or str(uuid.uuid4())
```

### 4. test_database.py (新規)

データベース機能の包括的なテストスクリプト。

**テスト内容**:
1. DB初期化
2. チャット履歴の保存・取得
3. ドキュメント情報の保存・取得
4. RAG検索履歴の保存・取得
5. 統計情報の取得
6. データ削除

## データフロー

### チャット時のデータフロー

```
1. ユーザーがメッセージ送信
   ↓
2. session_id生成（または既存IDを使用）
   ↓
3. RAG使用時
   ├─→ RAGManager.search()
   │   ├─→ FAISSでベクトル検索
   │   └─→ DBに検索履歴を保存
   └─→ 検索結果を応答に反映
   ↓
4. DBManager.save_chat()でチャット履歴を保存
   ↓
5. レスポンスをクライアントに返す
```

### ドキュメントアップロード時のデータフロー

```
1. ユーザーがファイルをアップロード
   ↓
2. RAGManager.process_document()
   ├─→ ファイル読み込み
   ├─→ テキスト分割
   ├─→ FAISS ベクトルストアに保存
   └─→ DBManager.save_document()でメタデータを保存
   ↓
3. アップロード完了レスポンス
```

## 統計情報

`/statistics` エンドポイントで取得できる情報:

```json
{
  "total_chats": 100,
  "total_documents": 5,
  "total_rag_searches": 50,
  "rag_enabled_chats": 30,
  "unique_sessions": 10
}
```

## パフォーマンス最適化

### インデックス

検索パフォーマンス向上のため、以下のインデックスを作成:

```sql
CREATE INDEX idx_chat_session ON chat_history(session_id);
CREATE INDEX idx_chat_timestamp ON chat_history(timestamp);
CREATE INDEX idx_rag_timestamp ON rag_searches(timestamp);
```

### row_factory設定

SQLiteの結果を辞書形式で取得:

```python
self.conn.row_factory = sqlite3.Row
```

## 拡張性

### 将来の拡張案

1. **ユーザー管理**
   - `users` テーブルの追加
   - セッションとユーザーの紐付け

2. **会話コンテキスト管理**
   - 過去のチャット履歴を自動的にコンテキストに含める
   - セッション別のRAGフィルタリング

3. **分析機能**
   - チャット傾向の分析
   - よく使われるモデルの統計
   - RAG使用率の推移

4. **エクスポート機能**
   - チャット履歴のCSV/JSONエクスポート
   - データバックアップ機能

## ベストプラクティス

### データベース接続管理

```python
# check_same_thread=False でスレッドセーフに
self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
```

### エラーハンドリング

```python
try:
    db_manager.save_chat(...)
except Exception as e:
    logger.error(f"DB保存エラー: {e}")
    # エラーでもアプリを停止させない
```

### トランザクション管理

```python
cursor.execute("INSERT ...")
self.conn.commit()  # 各操作後にコミット
```

## トラブルシューティング

### データベースロックエラー

**原因**: 複数スレッドからの同時アクセス

**解決**:
```python
sqlite3.connect(db_path, check_same_thread=False)
```

### メモリ使用量

**問題**: 大量のチャット履歴でメモリ消費

**解決**:
```python
# limitパラメータで取得件数を制限
get_chat_history(session_id, limit=50)
```

### JSON保存エラー

**問題**: retrieved_docsのシリアライズエラー

**解決**:
```python
json.dumps(retrieved_docs, ensure_ascii=False)
```

## テスト方法

### 単体テスト

```bash
cd backend
python test_database.py
```

### 統合テスト

1. FastAPIサーバー起動
2. チャット送信
3. `/statistics` エンドポイントで確認

```bash
curl http://localhost:8000/statistics
```

## まとめ

この実装により、以下が実現されました:

1. **永続化**: チャット履歴とドキュメント情報の永続的な保存
2. **分析**: 統計情報と人気クエリの分析
3. **スケーラビリティ**: SQLite + FAISSのハイブリッド構成による効率的なデータ管理
4. **拡張性**: 将来の機能追加に対応可能な設計

**技術スタック**:
- SQLite: 軽量で導入が簡単
- FAISS: 高速ベクトル検索
- Python sqlite3: 標準ライブラリで追加インストール不要
