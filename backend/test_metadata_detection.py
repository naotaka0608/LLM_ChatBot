"""
メタデータのみの要求検出をテストするスクリプト
"""

import sys
import io

# 文字コード問題を回避
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# テストケース
test_cases = [
    {
        "message": "アップロードしたファイル名を列挙して。内容は表示しないで。",
        "expected": True,
        "description": "ファイル名列挙 + 内容は表示しない"
    },
    {
        "message": "ファイル一覧を見せて。内容を表示しないで。",
        "expected": True,
        "description": "ファイル一覧 + 内容を表示しない"
    },
    {
        "message": "ドキュメントのリストを教えて。ファイル名だけでいいです。",
        "expected": True,
        "description": "リスト + ファイル名だけ"
    },
    {
        "message": "アップロードしたファイルの一覧をファイル名のみで表示して。",
        "expected": True,
        "description": "一覧 + ファイル名のみ"
    },
    {
        "message": "チームラボの合計金額は？",
        "expected": False,
        "description": "通常のRAG検索クエリ"
    },
    {
        "message": "サトウさんはどんな人？",
        "expected": False,
        "description": "通常のRAG検索クエリ"
    },
    {
        "message": "ファイル名を教えて",
        "expected": True,
        "description": "ファイル名系 + アクション（教えて）"
    },
    {
        "message": "アップロードしたファイル名を列挙して",
        "expected": True,
        "description": "ファイル名系 + アクション（列挙）のみ"
    },
    {
        "message": "ファイル一覧を見せて",
        "expected": True,
        "description": "ファイル名系 + アクション（見せて）"
    }
]

# 検出ロジック（main.pyと同じ）
metadata_list_keywords = ["ファイル名", "ファイル一覧", "ドキュメント一覧"]
action_keywords = ["列挙", "教えて", "見せて", "表示", "リスト"]
no_content_keywords = ["内容は表示しない", "内容を表示しない", "ファイル名だけ", "ファイル名のみ"]

print("=" * 70)
print("メタデータのみ要求の検出テスト")
print("=" * 70)

passed = 0
failed = 0

for i, test in enumerate(test_cases, 1):
    message = test["message"]
    expected = test["expected"]
    description = test["description"]

    # 検出ロジック
    has_metadata_keyword = any(keyword in message for keyword in metadata_list_keywords)
    has_action_keyword = any(keyword in message for keyword in action_keywords)
    has_no_content_keyword = any(keyword in message for keyword in no_content_keywords)

    is_metadata_only = (
        (has_metadata_keyword and has_action_keyword) or
        (has_metadata_keyword and has_no_content_keyword) or
        has_no_content_keyword
    )

    # 結果判定
    success = is_metadata_only == expected
    status = "✓ PASS" if success else "✗ FAIL"

    if success:
        passed += 1
    else:
        failed += 1

    print(f"\n[{i}] {status}")
    print(f"    クエリ: \"{message}\"")
    print(f"    期待値: {expected}, 実際: {is_metadata_only}")
    print(f"    説明: {description}")

print("\n" + "=" * 70)
print(f"テスト結果: {passed}件成功 / {failed}件失敗 / 全{len(test_cases)}件")
print("=" * 70)

if failed == 0:
    print("\n✓ すべてのテストに合格しました！")
else:
    print(f"\n⚠ {failed}件のテストが失敗しました")
