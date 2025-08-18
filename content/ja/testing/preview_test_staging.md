---
title: ステージングブランチでのプレビューテスト
type: docs
url: /ja/testing/:filename
menu:
  testing:
    identifier: ja-testing-preview_test_staging
---

# プレビューリンク生成テスト（ステージング）

このページは、`preview-links-staging` ブランチに対するPRで、日本語コンテンツのプレビューリンクが正しく生成されることを確認するためのテストページです。

## このテストの目的

- `preview-links-staging` ブランチの更新されたGitHub Actionが正しく動作することを確認
- 日本語の `pageurls.json` が正しく読み込まれることを確認
- プレビューリンクが生成されることを確認

## 期待される結果

このPRのコメントに、このページへのプレビューリンクが表示されるはずです。