---
title: Manage artifact storage and memory allocation
description: W&B アーティファクトのストレージやメモリ割り当てを管理する。
menu:
  default:
    identifier: ja-guides-core-artifacts-manage-data-storage
    parent: manage-data
---

W&B は、アーティファクトファイルをアメリカ合衆国にあるプライベートな Google Cloud Storage バケットにデフォルトで保存します。すべてのファイルは保存時および転送時に暗号化されます。

機密性の高いファイルについては、[Private Hosting]({{< relref path="/guides/hosting/" lang="ja" >}}) を設定するか、[reference artifacts]({{< relref path="../track-external-files.md" lang="ja" >}}) を使用することをお勧めします。

トレーニング中、W&B は以下のローカルディレクトリにログ、アーティファクト、および設定ファイルをローカルに保存します:

| ファイル   | デフォルトの場所  | デフォルトの場所を変更するには:                                    |
| --------- | ----------------- | --------------------------------------------------------------- |
| logs      | `./wandb`         | `wandb.init` の `dir` または `WANDB_DIR` 環境変数を設定         |
| artifacts | `~/.cache/wandb`  | `WANDB_CACHE_DIR` 環境変数を設定                                |
| configs   | `~/.config/wandb` | `WANDB_CONFIG_DIR` 環境変数を設定                               |

{{% alert color="secondary" %}}
`wandb` が初期化されるマシンによっては、デフォルトのフォルダがファイルシステムの書き込み可能な部分にないことがあります。これによりエラーが発生する可能性があります。
{{% /alert %}}

### ローカルのアーティファクトキャッシュをクリーンアップする

W&B は、共通ファイルを共有するバージョン間でのダウンロード速度を向上させるためにアーティファクトファイルをキャッシュします。時間が経つと、このキャッシュディレクトリは大きくなる可能性があります。キャッシュを削除し、最近使用されていないファイルを削除するには、[`wandb artifact cache cleanup`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-cache/" lang="ja" >}}) コマンドを実行します。

以下のコードスニペットは、キャッシュのサイズを 1GB に制限する方法を示しています。このコードスニペットをターミナルにコピー＆ペーストしてください:

```bash
$ wandb artifact cache cleanup 1GB
```