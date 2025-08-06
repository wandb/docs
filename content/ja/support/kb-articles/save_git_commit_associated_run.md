---
title: 自分の run に関連付けられた git コミットを保存するにはどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
---

`wandb.init` を呼び出すと、システムは自動的に git 情報（リモートリポジトリのリンクや最新コミットの SHA など）を収集します。この情報は [run ページ]({{< relref "/guides/models/track/runs/#view-logged-runs" >}}) に表示されます。この情報を表示するには、スクリプトを実行する際のカレントディレクトリーが git 管理下のフォルダーであることを確認してください。

git コミットや実験を実行したコマンドはユーザーには表示されますが、外部ユーザーからは見えません。パブリックプロジェクトでは、これらの詳細は非公開のままです。