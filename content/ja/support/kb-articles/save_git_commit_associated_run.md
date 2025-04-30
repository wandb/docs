---
title: run に関連する git commit をどのように保存できますか？
menu:
  support:
    identifier: ja-support-kb-articles-save_git_commit_associated_run
support:
- 実験管理
toc_hide: true
type: docs
url: /ja/support/:filename
---

`wandb.init` が呼び出されると、システムはリモートリポジトリのリンクや最新のコミットの SHA を含む git 情報を自動的に収集します。この情報は [runs ページ]({{< relref path="/guides/models/track/runs/#view-logged-runs" lang="ja" >}}) に表示されます。スクリプトを実行する際は、作業しているディレクトリーが git 管理フォルダー内であることを確認して、この情報を表示します。

git コミットと、実験を実行するために使用したコマンドはユーザーには見えますが、外部ユーザーからは隠されています。公開プロジェクトでは、これらの詳細は非公開のままです。