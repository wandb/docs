---
title: run に関連付けられた git commit を保存するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-save_git_commit_associated_run
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init` を実行すると、システムはリモートリポジトリのリンクや最新コミットの SHA など、git の情報を自動的に収集します。この情報は [run ページ]({{< relref path="/guides/models/track/runs/#view-logged-runs" lang="ja" >}}) に表示されます。この情報を見るためには、スクリプトを実行する際のカレントディレクトリーが git 管理下のフォルダであることを確認してください。

git のコミット情報や実験を実行したコマンドはユーザーには表示されますが、外部のユーザーからは非表示となります。公開プロジェクトの場合でも、これらの詳細はプライベートなままです。