---
title: アーティファクトのストレージとメモリの割り当てを管理する
description: W&B アーティファクトのストレージやメモリ割り当てを管理します。
menu:
  default:
    identifier: ja-guides-core-artifacts-manage-data-storage
    parent: manage-data
---

W&B は、アーティファクトファイルを米国にある Google Cloud Storage のプライベートバケットにデフォルトで保存します。すべてのファイルは、静止時および転送中に暗号化されています。

機密性の高いファイルには、[プライベートホスティング]({{< relref path="/guides/hosting/" lang="ja" >}})の設定や[参照アーティファクト]({{< relref path="../track-external-files.md" lang="ja" >}})の使用をお勧めします。

トレーニング中、W&B はログ、アーティファクト、および設定ファイルを以下のローカルディレクトリーにローカル保存します：

| File | Default location | To change default location set: |
| ---- | ---------------- | ------------------------------- |
| logs | `./wandb` | `wandb.init` の `dir` または `WANDB_DIR` 環境変数を設定 |
| artifacts | `~/.cache/wandb` | `WANDB_CACHE_DIR` 環境変数を設定 |
| configs | `~/.config/wandb` | `WANDB_CONFIG_DIR` 環境変数を設定 |
| ステージング用アーティファクトのアップロード | `~/.cache/wandb-data/` | `WANDB_DATA_DIR` 環境変数を設定 |
| ダウンロードされたアーティファクト | `./artifacts` | `WANDB_ARTIFACT_DIR` 環境変数を設定 |

W&B を設定するための環境変数の完全なガイドについては、[環境変数リファレンス]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})を参照してください。

{{% alert color="secondary" %}}
`wandb` が初期化されたマシンによっては、これらのデフォルトフォルダーがファイルシステムの書き込み可能な部分にない場合があります。これによりエラーが発生する可能性があります。
{{% /alert %}}

### ローカルのアーティファクトキャッシュをクリーンアップする

W&B は、共通ファイルを共有するバージョン間でダウンロードを高速化するためにアーティファクトファイルをキャッシュします。時間の経過とともに、このキャッシュディレクトリーは大きくなる可能性があります。キャッシュを整理し、最近使用されていないファイルを削除するために、[`wandb artifact cache cleanup`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-cache/" lang="ja" >}}) コマンドを実行してください。

以下のコードスニペットは、キャッシュサイズを1GBに制限する方法を示しています。コードスニペットをコピーしてターミナルに貼り付けてください：

```bash
$ wandb artifact cache cleanup 1GB
```