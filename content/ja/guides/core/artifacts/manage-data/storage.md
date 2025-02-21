---
title: Manage artifact storage and memory allocation
description: W&B Artifacts のストレージ、メモリアロケーションを管理します。
menu:
  default:
    identifier: ja-guides-core-artifacts-manage-data-storage
    parent: manage-data
---

W&B は、デフォルトで米国にあるプライベートな Google Cloud Storage バケットに Artifact のファイルを保存します。すべてのファイルは、保存時および転送中に暗号化されます。

機密性の高いファイルについては、[プライベートホスティング]({{< relref path="/guides/hosting/" lang="ja" >}}) を設定するか、[参照 Artifact ]({{< relref path="../track-external-files.md" lang="ja" >}}) を使用することをお勧めします。

トレーニング中、W&B はログ、Artifact 、および設定ファイルを次のローカルディレクトリーにローカルに保存します。

| ファイル | デフォルトの場所 | デフォルトの場所を変更するには、以下を設定します。|
| --- | --- | --- |
| ログ | `./wandb` | `wandb.init` の `dir` 、または `WANDB_DIR` 環境変数を設定します。|
| Artifact | `~/.cache/wandb` | `WANDB_CACHE_DIR` 環境変数 |
| 設定 | `~/.config/wandb` | `WANDB_CONFIG_DIR` 環境変数 |

{{% alert color="secondary" %}}
`wandb` が初期化されるマシンによっては、これらのデフォルトフォルダーがファイルシステムの書き込み可能な部分に配置されていない場合があります。これにより、エラーが発生する可能性があります。
{{% /alert %}}

### ローカル Artifact キャッシュのクリーンアップ

W&B は、ファイルを共有する バージョン 間でダウンロードを高速化するために、 Artifact ファイルをキャッシュします。時間の経過とともに、このキャッシュディレクトリーが大きくなる可能性があります。[`wandb artifact cache cleanup`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-cache/" lang="ja" >}}) コマンドを実行して、キャッシュを整理し、最近使用されていないファイルを削除します。

次のコードスニペットは、キャッシュのサイズを 1GB に制限する方法を示しています。コードスニペットをコピーして ターミナル に貼り付けます。

```bash
$ wandb artifact cache cleanup 1GB
```
