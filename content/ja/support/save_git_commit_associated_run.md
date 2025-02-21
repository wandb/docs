---
title: How can I save the git commit associated with my run?
menu:
  support:
    identifier: ja-support-save_git_commit_associated_run
tags:
- experiments
toc_hide: true
type: docs
---

`wandb.init` が呼び出されると、システムは自動的に git の情報（リモートリポジトリーのリンクや最新コミットの SHA など）を収集します。この情報は、[run ページ]({{< relref path="/guides/models/track/runs/#view-logged-runs" lang="ja" >}}) に表示されます。この情報を表示するには、スクリプトを実行する際の現在の作業ディレクトリーが、git で管理されたフォルダー内にあることを確認してください。

git のコミットと experiment の実行に使用された コマンド は、 ユーザー には表示されますが、外部 ユーザー には表示されません。パブリックの プロジェクト では、これらの詳細は非公開のままです。
