---
title: run に関連付けられている git commit を保存するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-save_git_commit_associated_run
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init` を呼び出すと、システムはリモートリポジトリーのリンクや最新コミットの SHA などの git の情報を自動的に収集します。この情報は [run ページ]({{< relref path="/guides/models/track/runs/#view-logged-runs" lang="ja" >}}) に表示されます。この情報を表示するには、スクリプトを実行する際の作業ディレクトリーが git 管理下にあることを確認してください。

git のコミットと実験の実行に用いた コマンドは ユーザーには表示されますが、外部の ユーザーからは非表示です。public Projects でも、これらの詳細は非公開のままです。