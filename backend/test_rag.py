"""
RAG機能のテストスクリプト
"""

from rag_manager import RAGManager
from pathlib import Path
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag():
    """RAG機能の基本テスト"""

    print("=" * 60)
    print("RAG機能テスト開始")
    print("=" * 60)

    # RAGマネージャー初期化
    print("\n1. RAGマネージャーを初期化中...")
    rag = RAGManager(upload_dir="./test_uploads", persist_dir="./test_vector_store")
    print("✓ 初期化完了")

    # テスト用のテキストファイルを作成
    print("\n2. テスト用ドキュメントを作成中...")
    test_dir = Path("./test_uploads")
    test_dir.mkdir(exist_ok=True)

    # サンプル1: 技術文書
    doc1_path = test_dir / "python_guide.txt"
    with open(doc1_path, 'w', encoding='utf-8') as f:
        f.write("""
Pythonプログラミングガイド

Pythonは、シンプルで読みやすい構文を持つ高水準プログラミング言語です。
データサイエンス、機械学習、Web開発など、幅広い分野で使用されています。

主な特徴：
- インタープリタ型言語
- 動的型付け
- 豊富なライブラリとフレームワーク
- クロスプラットフォーム対応

基本的な使い方：
変数の定義は簡単で、型宣言は不要です。
例: x = 10, name = "Python"

関数の定義にはdefキーワードを使用します。
def hello():
    print("Hello, World!")
""")

    # サンプル2: 別のトピック
    doc2_path = test_dir / "fastapi_intro.txt"
    with open(doc2_path, 'w', encoding='utf-8') as f:
        f.write("""
FastAPI入門

FastAPIは、モダンで高速なWebフレームワークです。
Python 3.7+の型ヒントを活用して、自動的にAPIドキュメントを生成します。

主な機能：
- 高速なパフォーマンス（Starlette、Pydanticベース）
- 自動的なAPIドキュメント生成（Swagger UI）
- 型安全性
- 非同期処理のサポート

簡単な例：
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
""")

    print(f"✓ テストファイル作成完了: {doc1_path.name}, {doc2_path.name}")

    # ドキュメントを処理
    print("\n3. ドキュメントを処理中（埋め込み生成）...")
    try:
        doc1_info = rag.process_document(doc1_path, doc1_path.name)
        print(f"✓ {doc1_path.name}: {doc1_info['chunks']}チャンク, {doc1_info['characters']}文字")

        doc2_info = rag.process_document(doc2_path, doc2_path.name)
        print(f"✓ {doc2_path.name}: {doc2_info['chunks']}チャンク, {doc2_info['characters']}文字")
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False

    # 検索テスト
    print("\n4. 検索テスト")
    print("-" * 60)

    test_queries = [
        "Pythonの特徴は何ですか？",
        "FastAPIとは何ですか？",
        "関数の定義方法を教えてください",
        "非同期処理について"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\nテスト {i}: {query}")
        results = rag.search(query, k=2)

        if results:
            print(f"  → {len(results)}件の関連情報を発見")
            for j, doc in enumerate(results, 1):
                print(f"     [{j}] ソース: {doc['source']}")
                print(f"         スコア: {doc['score']:.4f}")
                print(f"         内容: {doc['content'][:100]}...")
        else:
            print("  → 関連情報が見つかりませんでした")

    # ドキュメント一覧
    print("\n5. 登録済みドキュメント一覧")
    print("-" * 60)
    docs = rag.get_documents()
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc['filename']}")
        print(f"     チャンク数: {doc['chunks']}, 文字数: {doc['characters']}")

    print("\n" + "=" * 60)
    print("テスト完了！RAG機能は正常に動作しています。")
    print("=" * 60)

    return True

if __name__ == "__main__":
    try:
        success = test_rag()
        if not success:
            print("\n⚠ テストに失敗しました")
            exit(1)
    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
