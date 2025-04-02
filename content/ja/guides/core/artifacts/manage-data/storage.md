---
title: Manage artifact storage and memory allocation
description: W&B Artifacts のストレージ、メモリアロケーションを管理します。
menu:
  default:
    identifier: ja-guides-core-artifacts-manage-data-storage
    parent: manage-data
---

W&B は artifact ファイルを、デフォルトで米国にあるプライベートな Google Cloud Storage バケットに保存します。すべてのファイルは、保存時および転送時に暗号化されます。

機密性の高いファイルについては、[Private Hosting]({{< relref path="/guides/hosting/" lang="ja" >}}) をセットアップするか、[reference artifacts]({{< relref path="../track-external-files.md" lang="ja" >}}) を使用することをお勧めします。

トレーニング中、W&B はログ、アーティファクト、および設定ファイルを次のローカルディレクトリーにローカルに保存します。

| ファイル | デフォルトの場所 | デフォルトの場所を変更するには、以下を設定します： |
| ---- | ---------------- | ------------------------------- |
| ログ | `./wandb` | `wandb.init` の `dir` 、または `WANDB_DIR` 環境変数を設定 |
| artifacts | `~/.cache/wandb` | `WANDB_CACHE_DIR` 環境変数 |
| configs | `~/.config/wandb` | `WANDB_CONFIG_DIR` 環境変数 |
| アップロード用に artifacts をステージング | `~/.cache/wandb-data/` | `WANDB_DATA_DIR` 環境変数 |
| ダウンロードされた artifacts | `./artifacts` | `WANDB_ARTIFACT_DIR` 環境変数 |

環境変数を使用して W&B を構成するための完全なガイドについては、[環境変数のリファレンス]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})を参照してください。

{{% alert color="secondary" %}}
`wandb` が初期化されるマシンによっては、これらのデフォルトフォルダーがファイルシステムの書き込み可能な場所に配置されていない場合があります。これにより、エラーが発生する可能性があります。
{{% /alert %}}

### ローカル artifact キャッシュのクリーンアップ

W&B は artifact ファイルをキャッシュして、ファイルを共有する バージョン 間でのダウンロードを高速化します。時間の経過とともに、このキャッシュディレクトリーが大きくなる可能性があります。[`wandb artifact cache cleanup`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-cache/" lang="ja" >}}) コマンドを実行して、キャッシュを削除し、最近使用されていないファイルを削除します。

次のコードスニペットは、キャッシュのサイズを 1GB に制限する方法を示しています。コードスニペットをコピーして ターミナル に貼り付けます。

```bash
$ wandb artifact cache cleanup 1GB
```
