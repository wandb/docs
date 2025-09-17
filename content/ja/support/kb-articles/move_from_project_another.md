---
title: run を 別のプロジェクトに移動することは可能ですか？
menu:
  support:
    identifier: ja-support-kb-articles-move_from_project_another
support:
- runs
toc_hide: true
type: docs
url: /support/:filename
---

次の手順に従うと、run を 別の プロジェクト に 移動できます:

- 移動したい run が ある プロジェクト ページ に 移動します。
- **Runs** タブ を クリックして、Runs テーブル を 開きます。
- 移動する run を 選択します。
- **Move** ボタン を クリックします。
- 移動先 の プロジェクト を 選択し、操作 を 確認します。

W&B は UI からの run の 移動 を サポートしていますが、run の コピー は サポートしていません。run と 一緒に ログ された Artifacts は 新しい プロジェクト へ は 移動されません。Artifacts を run の 新しい 場所 に 手動で 移す には、[`wandb artifact get`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-get/" lang="ja" >}}) の SDK コマンド または [`Api.artifact` API]({{< relref path="/ref/python/public-api/api/#artifact" lang="ja" >}}) を 使って artifact を ダウンロード し、続いて [wandb artifact put]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put/" lang="ja" >}}) または `Api.artifact` API を 使って run の 新しい 場所 に アップロード します。