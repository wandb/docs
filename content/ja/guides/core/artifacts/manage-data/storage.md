---
title: Artifacts のストレージとメモリ割り当てを管理する
description: W&B Artifacts のストレージとメモリの割り当てを管理します。
menu:
  default:
    identifier: ja-guides-core-artifacts-manage-data-storage
    parent: manage-data
---

W&B は、デフォルトで米国に所在するプライベートな Google Cloud Storage バケットにアーティファクトファイルを保存します。すべてのファイルは、保存時と転送時に暗号化されます。
機密性の高いファイルについては、[プライベートホスティング]({{< relref path="/guides/hosting/" lang="ja" >}})を設定するか、[参照アーティファクト]({{< relref path="../track-external-files.md" lang="ja" >}})を使用することをおすすめします。
トレーニング中、W&B はログ、アーティファクト、および設定ファイルを以下のローカルディレクトリーに保存します。
| ファイル | デフォルトの場所 | デフォルトの場所を変更するには、以下を設定します: |
| ---- | ---------------- | ----------------------------------- |
| ログ | `./wandb` | `wandb.init` で `dir` を設定するか、`WANDB_DIR` 環境変数を設定します |
| アーティファクト | `~/.cache/wandb` | `WANDB_CACHE_DIR` 環境変数 |
| 設定 | `~/.config/wandb` | `WANDB_CONFIG_DIR` 環境変数 |
| アップロードのためのステージングアーティファクト | `~/.cache/wandb-data/` | `WANDB_DATA_DIR` 環境変数 |
| ダウンロードされたアーティファクト | `./artifacts` | `WANDB_ARTIFACT_DIR` 環境変数 |
W&B を設定するための環境変数の完全なガイドについては、[環境変数リファレンス]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})を参照してください。
{{% alert color="secondary" %}}
`wandb` が初期化されるマシンによっては、これらのデフォルトフォルダーがファイルシステムの書き込み可能な部分にない場合があります。これによりエラーが発生する可能性があります。
{{% /alert %}}
### ローカルのアーティファクトキャッシュのクリーンアップ
W&B は、ファイルを共有するバージョン間でのダウンロードを高速化するために、アーティファクトファイルをキャッシュします。時間が経つと、このキャッシュディレクトリーは大きくなる可能性があります。[`wandb artifact cache cleanup`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-cache/" lang="ja" >}}) コマンドを実行して、キャッシュを整理し、最近使用されていないファイルを削除します。
次のコードスニペットは、キャッシュのサイズを 1 GB に制限する方法を示しています。このコードスニペットをターミナルにコピーして貼り付けます。
```bash
$ wandb artifact cache cleanup 1GB
```