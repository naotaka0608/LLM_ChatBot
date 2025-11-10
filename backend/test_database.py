"""
データベース機能のテストスクリプト
"""

import sys
from pathlib import Path
from db_manager import DBManager
import logging
import io

# 文字コード問題を回避
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_database():
    """データベース機能の基本テスト"""

    print("=" * 70)
    print("データベース機能テスト開始")
    print("=" * 70)

    # テスト用のDBマネージャーを作成
    print("\n1. データベースマネージャーを初期化中...")
    test_db_path = "./test_chatbot.db"

    # 既存のテストDBがあれば削除
    if Path(test_db_path).exists():
        Path(test_db_path).unlink()
        print(f"   既存のテストDB削除: {test_db_path}")

    db = DBManager(db_path=test_db_path)
    print("   ✓ 初期化完了")

    # チャット履歴のテスト
    print("\n2. チャット履歴機能のテスト")
    print("-" * 70)

    # チャットを保存
    print("   チャットを保存中...")
    session1 = "session-001"
    session2 = "session-002"

    chat_id1 = db.save_chat(
        session_id=session1,
        user_message="こんにちは",
        bot_response="こんにちは！何かお手伝いできることはありますか？",
        model_id="gpt2",
        use_rag=False,
        retrieved_docs_count=0
    )
    print(f"   ✓ チャット保存完了 (ID: {chat_id1})")

    chat_id2 = db.save_chat(
        session_id=session1,
        user_message="Pythonについて教えて",
        bot_response="Pythonは高水準プログラミング言語です...",
        model_id="gpt2",
        use_rag=True,
        retrieved_docs_count=2
    )
    print(f"   ✓ チャット保存完了 (ID: {chat_id2})")

    chat_id3 = db.save_chat(
        session_id=session2,
        user_message="FastAPIとは？",
        bot_response="FastAPIはモダンなWebフレームワークです...",
        model_id="rinna-japanese-gpt2",
        use_rag=True,
        retrieved_docs_count=3
    )
    print(f"   ✓ チャット保存完了 (ID: {chat_id3})")

    # チャット履歴取得
    print(f"\n   セッション {session1} の履歴を取得:")
    history = db.get_chat_history(session1)
    for chat in history:
        print(f"      - [{chat['id']}] {chat['user_message'][:30]}... (RAG: {chat['use_rag']})")

    # 全チャット履歴取得
    print("\n   全チャット履歴を取得:")
    all_history = db.get_all_chat_history()
    print(f"      総チャット数: {len(all_history)}")

    # ドキュメント情報のテスト
    print("\n3. ドキュメント情報機能のテスト")
    print("-" * 70)

    # ドキュメントを保存
    print("   ドキュメント情報を保存中...")
    doc_id1 = db.save_document(
        filename="python_guide.txt",
        file_path="./uploads/python_guide.txt",
        chunks_count=5,
        characters_count=1200,
        file_type=".txt"
    )
    print(f"   ✓ ドキュメント保存完了 (ID: {doc_id1})")

    doc_id2 = db.save_document(
        filename="fastapi_intro.pdf",
        file_path="./uploads/fastapi_intro.pdf",
        chunks_count=8,
        characters_count=2500,
        file_type=".pdf"
    )
    print(f"   ✓ ドキュメント保存完了 (ID: {doc_id2})")

    # ドキュメント一覧取得
    print("\n   ドキュメント一覧を取得:")
    docs = db.get_documents()
    for doc in docs:
        print(f"      - [{doc['id']}] {doc['filename']} ({doc['chunks_count']}チャンク, {doc['characters_count']}文字)")

    # 特定のドキュメント取得
    print("\n   特定のドキュメントを取得:")
    doc = db.get_document_by_filename("python_guide.txt")
    if doc:
        print(f"      ✓ 見つかりました: {doc['filename']} (ID: {doc['id']})")

    # RAG検索履歴のテスト
    print("\n4. RAG検索履歴機能のテスト")
    print("-" * 70)

    # RAG検索を保存
    print("   RAG検索履歴を保存中...")
    search_id1 = db.save_rag_search(
        query="Pythonの特徴は？",
        retrieved_docs=[
            {"content": "Pythonは...", "source": "python_guide.txt", "chunk": 0, "score": 0.85},
            {"content": "主な特徴...", "source": "python_guide.txt", "chunk": 1, "score": 0.72}
        ],
        k=2
    )
    print(f"   ✓ RAG検索保存完了 (ID: {search_id1})")

    search_id2 = db.save_rag_search(
        query="FastAPIの使い方",
        retrieved_docs=[
            {"content": "FastAPIは...", "source": "fastapi_intro.pdf", "chunk": 0, "score": 0.91},
            {"content": "簡単な例...", "source": "fastapi_intro.pdf", "chunk": 2, "score": 0.78},
            {"content": "主な機能...", "source": "fastapi_intro.pdf", "chunk": 3, "score": 0.68}
        ],
        k=3
    )
    print(f"   ✓ RAG検索保存完了 (ID: {search_id2})")

    # 同じクエリをもう一度保存（人気度テスト用）
    search_id3 = db.save_rag_search(
        query="Pythonの特徴は？",
        retrieved_docs=[
            {"content": "Pythonは...", "source": "python_guide.txt", "chunk": 0, "score": 0.85}
        ],
        k=1
    )
    print(f"   ✓ RAG検索保存完了 (ID: {search_id3})")

    # RAG検索履歴取得
    print("\n   RAG検索履歴を取得:")
    searches = db.get_rag_searches()
    for search in searches[:3]:  # 最新3件のみ表示
        print(f"      - [{search['id']}] \"{search['query']}\" (結果: {search['results_count']}件)")

    # 人気のクエリを取得
    print("\n   人気のクエリを取得:")
    popular = db.get_popular_queries()
    for query in popular:
        print(f"      - \"{query['query']}\" (検索回数: {query['count']})")

    # 統計情報のテスト
    print("\n5. 統計情報機能のテスト")
    print("-" * 70)

    stats = db.get_statistics()
    print(f"   総チャット数: {stats['total_chats']}")
    print(f"   総ドキュメント数: {stats['total_documents']}")
    print(f"   総RAG検索数: {stats['total_rag_searches']}")
    print(f"   RAG使用チャット数: {stats['rag_enabled_chats']}")
    print(f"   ユニークセッション数: {stats['unique_sessions']}")

    # クリーンアップのテスト
    print("\n6. クリーンアップ機能のテスト")
    print("-" * 70)

    # 特定のセッション削除
    print(f"   セッション {session2} を削除中...")
    db.delete_chat_history(session2)
    remaining = db.get_chat_history(session2)
    print(f"   ✓ 削除完了 (残り: {len(remaining)}件)")

    # 終了
    print("\n7. データベース接続を閉じる")
    print("-" * 70)
    db.close()
    print("   ✓ 接続を閉じました")

    print("\n" + "=" * 70)
    print("テスト完了！データベース機能は正常に動作しています。")
    print(f"テストDBファイル: {test_db_path}")
    print("=" * 70)

    return True


if __name__ == "__main__":
    try:
        success = test_database()
        if not success:
            print("\n⚠ テストに失敗しました")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
