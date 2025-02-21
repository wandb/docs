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

`wandb.init` が呼び出されると、システムはリモートリポジトリのリンクや最新のコミットの SHA を含む git の情報を自動的に収集します。この情報は [run ページ]({{< relref path="/guides/models/track/runs/#view-logged-runs" lang="ja" >}}) に表示されます。スクリプトを実行する際の作業ディレクトリーが git 管理フォルダ内であることを確認し、この情報を表示します。

git コミットと実験を実行するために使用されるコマンドは、ユーザーには表示されますが、外部のユーザーからは非表示になります。公開プロジェクトでは、これらの詳細は非公開のままです。