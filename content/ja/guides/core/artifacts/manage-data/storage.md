---
title: アーティファクトのストレージとメモリ割り当てを管理する
description: W&B Artifacts のストレージやメモリ割り当てを管理します。
menu:
  default:
    identifier: storage
    parent: manage-data
---

W&B は、アーティファクトファイルをデフォルトでアメリカ合衆国にあるプライベートな Google Cloud Storage バケットに保存します。すべてのファイルは保存時と転送時の両方で暗号化されます。

機密性の高いファイルについては、[Private Hosting]({{< relref "/guides/hosting/" >}}) のセットアップや [reference artifacts]({{< relref "../track-external-files.md" >}}) の利用をおすすめします。

トレーニング中、W&B はローカルでログ・Artifacts・設定ファイルを以下のローカルディレクトリーに保存します。

| ファイル | デフォルト保存場所 | デフォルト保存場所を変更する方法 |
| ---- | ---------------- | ------------------------------- |
| logs | `./wandb` | `wandb.init` の `dir` または `WANDB_DIR` 環境変数を設定 |
| artifacts | `~/.cache/wandb` | `WANDB_CACHE_DIR` 環境変数を設定 |
| configs | `~/.config/wandb` | `WANDB_CONFIG_DIR` 環境変数を設定 |
| staging artifacts for upload  | `~/.cache/wandb-data/` | `WANDB_DATA_DIR` 環境変数を設定 |
| downloaded artifacts | `./artifacts` | `WANDB_ARTIFACT_DIR` 環境変数を設定 |

W&B の設定に使える環境変数については、[環境変数リファレンス]({{< relref "/guides/models/track/environment-variables.md" >}}) のガイドを参照してください。

{{% alert color="secondary" %}}
`wandb` を初期化するマシンによっては、これらのデフォルトフォルダーがファイルシステム上で書き込み可能な場所にない場合があります。その場合、エラーが発生することがあります。
{{% /alert %}}

### ローカルの Artifact キャッシュをクリーンアップする

W&B は、バージョン間で共通するファイルのダウンロードを高速化するために Artifact ファイルをキャッシュしています。時間の経過とともに、このキャッシュディレクトリーは大きくなります。[`wandb artifact cache cleanup`]({{< relref "/ref/cli/wandb-artifact/wandb-artifact-cache/" >}}) コマンドを実行すると、キャッシュを整理し、最近使っていないファイルを削除できます。

以下のコードスニペットは、キャッシュサイズを 1GB に制限する方法を示しています。ターミナルにコピー＆ペーストしてご利用ください。

```bash
# キャッシュサイズを 1GB に制限してクリーンアップします
$ wandb artifact cache cleanup 1GB
```