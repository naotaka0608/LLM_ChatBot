"""
データベース管理モジュール
SQLiteを使用してチャット履歴、ドキュメント情報、RAG検索履歴、ユーザー情報を管理
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import logging
import hashlib
import secrets

logger = logging.getLogger(__name__)


class DBManager:
    """SQLiteデータベース管理クラス"""

    def __init__(self, db_path: str = "./chatbot.db"):
        """
        初期化

        Args:
            db_path: データベースファイルのパス
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_tables()
        logger.info(f"データベース初期化完了: {db_path}")

    def _connect(self):
        """データベースに接続"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # 辞書形式で結果を取得
            logger.info("データベース接続成功")
        except Exception as e:
            logger.error(f"データベース接続エラー: {e}")
            raise

    def _create_tables(self):
        """必要なテーブルを作成"""
        cursor = self.conn.cursor()

        # チャット履歴テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_id TEXT,
                use_rag BOOLEAN DEFAULT 0,
                retrieved_docs_count INTEGER DEFAULT 0
            )
        """)

        # ドキュメント情報テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                file_path TEXT NOT NULL,
                chunks_count INTEGER NOT NULL,
                characters_count INTEGER NOT NULL,
                upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                file_type TEXT
            )
        """)

        # RAG検索履歴テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                retrieved_docs TEXT,
                k INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                results_count INTEGER DEFAULT 0
            )
        """)

        # ユーザーテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME
            )
        """)

        # インデックス作成（検索パフォーマンス向上）
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_session
            ON chat_history(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_timestamp
            ON chat_history(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_timestamp
            ON rag_searches(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_username
            ON users(username)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_email
            ON users(email)
        """)

        self.conn.commit()
        logger.info("テーブル作成完了")

    # ==================== チャット履歴関連 ====================

    def save_chat(self, session_id: str, user_message: str, bot_response: str,
                  model_id: str = None, use_rag: bool = False,
                  retrieved_docs_count: int = 0) -> int:
        """
        チャット履歴を保存

        Args:
            session_id: セッションID
            user_message: ユーザーのメッセージ
            bot_response: ボットの応答
            model_id: 使用したモデルID
            use_rag: RAGを使用したか
            retrieved_docs_count: 取得したドキュメント数

        Returns:
            保存したレコードのID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history
            (session_id, user_message, bot_response, model_id, use_rag, retrieved_docs_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, user_message, bot_response, model_id, use_rag, retrieved_docs_count))

        self.conn.commit()
        chat_id = cursor.lastrowid
        logger.info(f"チャット履歴保存: ID={chat_id}, session={session_id}")
        return chat_id

    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """
        セッションのチャット履歴を取得

        Args:
            session_id: セッションID
            limit: 取得する最大件数

        Returns:
            チャット履歴のリスト
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM chat_history
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_all_chat_history(self, limit: int = 100) -> List[Dict]:
        """
        全てのチャット履歴を取得

        Args:
            limit: 取得する最大件数

        Returns:
            チャット履歴のリスト
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM chat_history
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def delete_chat_history(self, session_id: str = None):
        """
        チャット履歴を削除

        Args:
            session_id: 削除するセッションID（Noneの場合は全削除）
        """
        cursor = self.conn.cursor()
        if session_id:
            cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
            logger.info(f"セッション {session_id} の履歴を削除")
        else:
            cursor.execute("DELETE FROM chat_history")
            logger.info("全てのチャット履歴を削除")

        self.conn.commit()

    # ==================== ドキュメント関連 ====================

    def save_document(self, filename: str, file_path: str, chunks_count: int,
                      characters_count: int, file_type: str = None) -> int:
        """
        ドキュメント情報を保存

        Args:
            filename: ファイル名
            file_path: ファイルパス
            chunks_count: チャンク数
            characters_count: 文字数
            file_type: ファイルタイプ（拡張子）

        Returns:
            保存したレコードのID
        """
        cursor = self.conn.cursor()

        # 既存のドキュメントがあれば更新
        cursor.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
        existing = cursor.fetchone()

        if existing:
            cursor.execute("""
                UPDATE documents
                SET file_path = ?, chunks_count = ?, characters_count = ?,
                    file_type = ?, upload_date = CURRENT_TIMESTAMP
                WHERE filename = ?
            """, (file_path, chunks_count, characters_count, file_type, filename))
            doc_id = existing[0]
            logger.info(f"ドキュメント更新: {filename} (ID={doc_id})")
        else:
            cursor.execute("""
                INSERT INTO documents
                (filename, file_path, chunks_count, characters_count, file_type)
                VALUES (?, ?, ?, ?, ?)
            """, (filename, file_path, chunks_count, characters_count, file_type))
            doc_id = cursor.lastrowid
            logger.info(f"ドキュメント保存: {filename} (ID={doc_id})")

        self.conn.commit()
        return doc_id

    def get_documents(self) -> List[Dict]:
        """
        全てのドキュメント情報を取得

        Returns:
            ドキュメント情報のリスト
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM documents
            ORDER BY upload_date DESC
        """)

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_document_by_filename(self, filename: str) -> Optional[Dict]:
        """
        ファイル名でドキュメント情報を取得

        Args:
            filename: ファイル名

        Returns:
            ドキュメント情報（存在しない場合はNone）
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE filename = ?", (filename,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def delete_document(self, filename: str):
        """
        ドキュメント情報を削除

        Args:
            filename: 削除するファイル名
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM documents WHERE filename = ?", (filename,))
        self.conn.commit()
        logger.info(f"ドキュメント削除: {filename}")

    def delete_all_documents(self):
        """全てのドキュメント情報を削除"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM documents")
        self.conn.commit()
        logger.info("全てのドキュメント情報を削除")

    # ==================== RAG検索履歴関連 ====================

    def save_rag_search(self, query: str, retrieved_docs: List[Dict], k: int = 3) -> int:
        """
        RAG検索履歴を保存

        Args:
            query: 検索クエリ
            retrieved_docs: 取得したドキュメントのリスト
            k: 取得数

        Returns:
            保存したレコードのID
        """
        cursor = self.conn.cursor()

        # retrieved_docsをJSON文字列に変換
        docs_json = json.dumps(retrieved_docs, ensure_ascii=False)
        results_count = len(retrieved_docs)

        cursor.execute("""
            INSERT INTO rag_searches
            (query, retrieved_docs, k, results_count)
            VALUES (?, ?, ?, ?)
        """, (query, docs_json, k, results_count))

        self.conn.commit()
        search_id = cursor.lastrowid
        logger.info(f"RAG検索履歴保存: ID={search_id}, query='{query[:50]}...'")
        return search_id

    def get_rag_searches(self, limit: int = 50) -> List[Dict]:
        """
        RAG検索履歴を取得

        Args:
            limit: 取得する最大件数

        Returns:
            検索履歴のリスト
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM rag_searches
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        results = []
        for row in rows:
            row_dict = dict(row)
            # JSON文字列をPythonオブジェクトに変換
            if row_dict['retrieved_docs']:
                row_dict['retrieved_docs'] = json.loads(row_dict['retrieved_docs'])
            results.append(row_dict)

        return results

    def get_popular_queries(self, limit: int = 10) -> List[Dict]:
        """
        人気の検索クエリを取得（頻度順）

        Args:
            limit: 取得する最大件数

        Returns:
            クエリと出現回数のリスト
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT query, COUNT(*) as count
            FROM rag_searches
            GROUP BY query
            ORDER BY count DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def delete_rag_searches(self):
        """全てのRAG検索履歴を削除"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM rag_searches")
        self.conn.commit()
        logger.info("全てのRAG検索履歴を削除")

    # ==================== ユーザー認証関連 ====================

    def _hash_password(self, password: str) -> str:
        """
        パスワードをハッシュ化

        Args:
            password: 平文のパスワード

        Returns:
            ハッシュ化されたパスワード
        """
        return hashlib.sha256(password.encode()).hexdigest()

    def create_user(self, username: str, email: str, password: str) -> Optional[int]:
        """
        新規ユーザーを作成

        Args:
            username: ユーザー名
            email: メールアドレス
            password: パスワード

        Returns:
            作成されたユーザーID（失敗時はNone）
        """
        cursor = self.conn.cursor()
        password_hash = self._hash_password(password)

        try:
            cursor.execute("""
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
            """, (username, email, password_hash))
            self.conn.commit()
            user_id = cursor.lastrowid
            logger.info(f"新規ユーザー作成: {username} (ID={user_id})")
            return user_id
        except sqlite3.IntegrityError as e:
            logger.warning(f"ユーザー作成失敗: {e}")
            return None

    def verify_user(self, username: str, password: str) -> Optional[Dict]:
        """
        ユーザー認証

        Args:
            username: ユーザー名
            password: パスワード

        Returns:
            認証成功時はユーザー情報、失敗時はNone
        """
        cursor = self.conn.cursor()
        password_hash = self._hash_password(password)

        cursor.execute("""
            SELECT id, username, email, created_at
            FROM users
            WHERE username = ? AND password_hash = ?
        """, (username, password_hash))

        row = cursor.fetchone()
        if row:
            user_dict = dict(row)
            # 最終ログイン時刻を更新
            cursor.execute("""
                UPDATE users
                SET last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (user_dict['id'],))
            self.conn.commit()
            logger.info(f"ユーザー認証成功: {username}")
            return user_dict
        else:
            logger.warning(f"ユーザー認証失敗: {username}")
            return None

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """
        ユーザー名でユーザー情報を取得

        Args:
            username: ユーザー名

        Returns:
            ユーザー情報（存在しない場合はNone）
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, username, email, created_at, last_login
            FROM users
            WHERE username = ?
        """, (username,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """
        メールアドレスでユーザー情報を取得

        Args:
            email: メールアドレス

        Returns:
            ユーザー情報（存在しない場合はNone）
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, username, email, created_at, last_login
            FROM users
            WHERE email = ?
        """, (email,))
        row = cursor.fetchone()
        return dict(row) if row else None

    # ==================== 統計情報 ====================

    def get_statistics(self) -> Dict:
        """
        データベース全体の統計情報を取得

        Returns:
            統計情報の辞書
        """
        cursor = self.conn.cursor()

        # チャット数
        cursor.execute("SELECT COUNT(*) as count FROM chat_history")
        chat_count = cursor.fetchone()['count']

        # ドキュメント数
        cursor.execute("SELECT COUNT(*) as count FROM documents")
        doc_count = cursor.fetchone()['count']

        # RAG検索数
        cursor.execute("SELECT COUNT(*) as count FROM rag_searches")
        search_count = cursor.fetchone()['count']

        # RAG使用チャット数
        cursor.execute("SELECT COUNT(*) as count FROM chat_history WHERE use_rag = 1")
        rag_chat_count = cursor.fetchone()['count']

        # セッション数
        cursor.execute("SELECT COUNT(DISTINCT session_id) as count FROM chat_history")
        session_count = cursor.fetchone()['count']

        return {
            "total_chats": chat_count,
            "total_documents": doc_count,
            "total_rag_searches": search_count,
            "rag_enabled_chats": rag_chat_count,
            "unique_sessions": session_count
        }

    # ==================== クリーンアップ ====================

    def close(self):
        """データベース接続を閉じる"""
        if self.conn:
            self.conn.close()
            logger.info("データベース接続を閉じました")

    def __del__(self):
        """デストラクタ"""
        self.close()
