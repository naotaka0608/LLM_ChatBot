"""
RAGスコアフィルタリングのテストスクリプト
"""

import sys
import io
from pathlib import Path
from rag_manager import RAGManager
from db_manager import DBManager
import logging

# 文字コード問題を回避
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_score_filtering():
    """スコアフィルタリングのテスト"""

    print("=" * 70)
    print("RAGスコアフィルタリングテスト")
    print("=" * 70)

    # テスト用のRAGマネージャーを初期化
    print("\n1. RAGマネージャーを初期化中...")
    db_manager = DBManager(db_path="./test_chatbot.db")
    rag = RAGManager(
        upload_dir="./test_uploads",
        persist_dir="./test_vector_store",
        db_manager=db_manager
    )
    print("   ✓ 初期化完了")

    # テスト用ドキュメントが存在するか確認
    docs = rag.get_documents()
    if not docs:
        print("\n⚠ テスト用ドキュメントがありません")
        print("   先に test_rag.py を実行してドキュメントを作成してください")
        return False

    print(f"\n2. 登録済みドキュメント: {len(docs)}件")
    for doc in docs:
        print(f"   - {doc['filename']}")

    # テストクエリ
    test_query = "チームラボの合計金額は？"

    print(f"\n3. テストクエリ: \"{test_query}\"")
    print("-" * 70)

    # 閾値なし（全件取得）
    print("\n【テスト1】閾値なし（score_threshold=None）")
    results_no_threshold = rag.search(test_query, k=3, score_threshold=None)
    print(f"   結果件数: {len(results_no_threshold)}件")
    for i, doc in enumerate(results_no_threshold, 1):
        print(f"   [{i}] {doc['source']:<30} スコア: {doc['score']:.4f}")
        print(f"       内容: {doc['content'][:100]}...")

    # 閾値あり（厳しめ: 0.5）
    print("\n【テスト2】厳しめの閾値（score_threshold=0.5）")
    results_strict = rag.search(test_query, k=3, score_threshold=0.5)
    print(f"   結果件数: {len(results_strict)}件")
    for i, doc in enumerate(results_strict, 1):
        print(f"   [{i}] {doc['source']:<30} スコア: {doc['score']:.4f}")
        print(f"       内容: {doc['content'][:100]}...")

    # 閾値あり（標準: 1.0）
    print("\n【テスト3】標準的な閾値（score_threshold=1.0）")
    results_standard = rag.search(test_query, k=3, score_threshold=1.0)
    print(f"   結果件数: {len(results_standard)}件")
    for i, doc in enumerate(results_standard, 1):
        print(f"   [{i}] {doc['source']:<30} スコア: {doc['score']:.4f}")
        print(f"       内容: {doc['content'][:100]}...")

    # 閾値あり（緩め: 1.5）
    print("\n【テスト4】緩めの閾値（score_threshold=1.5）")
    results_loose = rag.search(test_query, k=3, score_threshold=1.5)
    print(f"   結果件数: {len(results_loose)}件")
    for i, doc in enumerate(results_loose, 1):
        print(f"   [{i}] {doc['source']:<30} スコア: {doc['score']:.4f}")
        print(f"       内容: {doc['content'][:100]}...")

    # 結果まとめ
    print("\n" + "=" * 70)
    print("テスト結果サマリー")
    print("=" * 70)
    print(f"閾値なし:          {len(results_no_threshold)}件")
    print(f"厳しめ (0.5):      {len(results_strict)}件")
    print(f"標準的 (1.0):      {len(results_standard)}件")
    print(f"緩め (1.5):        {len(results_loose)}件")

    print("\n推奨設定:")
    if len(results_standard) > 0:
        print("   ✓ score_threshold=1.0 が適切です")
        print("   　関連性の高い結果のみを返せています")
    elif len(results_loose) > 0:
        print("   ⚠ score_threshold=1.5 を推奨")
        print("   　少し緩めの閾値が必要かもしれません")
    else:
        print("   ⚠ 閾値なしまたは2.0以上を推奨")
        print("   　埋め込みモデルの精度を確認してください")

    print("\n" + "=" * 70)
    print("テスト完了！")
    print("=" * 70)

    return True


if __name__ == "__main__":
    try:
        success = test_score_filtering()
        if not success:
            print("\n⚠ テストに失敗しました")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
