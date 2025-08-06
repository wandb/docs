---
title: アーティファクトのストレージとメモリ割り当てを管理する
description: W&B Artifacts のストレージやメモリ割り当てを管理する方法について説明します。
menu:
  default:
    identifier: ja-guides-core-artifacts-manage-data-storage
    parent: manage-data
---

W&B は、アーティファクトファイルをデフォルトで米国内にあるプライベートな Google Cloud Storage バケットに保存します。すべてのファイルは保存時および転送時に暗号化されます。

機密性の高いファイルについては、[Private Hosting]({{< relref path="/guides/hosting/" lang="ja" >}}) のセットアップや [reference artifacts]({{< relref path="../track-external-files.md" lang="ja" >}}) の利用をおすすめします。

トレーニング中、W&B はログ、Artifacts、設定ファイルを以下のローカルディレクトリーに保存します:

| ファイル | デフォルト保存場所 | デフォルトの場所を変更するには |
| ---- | ---------------- | ------------------------------- |
| logs | `./wandb` | `wandb.init` の `dir` または `WANDB_DIR` 環境変数を設定 |
| artifacts | `~/.cache/wandb` | `WANDB_CACHE_DIR` 環境変数を設定 |
| configs | `~/.config/wandb` | `WANDB_CONFIG_DIR` 環境変数を設定 |
| アップロード用一時 artifacts | `~/.cache/wandb-data/` | `WANDB_DATA_DIR` 環境変数を設定 |
| ダウンロード済み artifacts | `./artifacts` | `WANDB_ARTIFACT_DIR` 環境変数を設定 |

W&B の設定に環境変数を使う方法の詳細は、[環境変数リファレンス]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をご覧ください。

{{% alert color="secondary" %}}
`wandb` の初期化を行うマシンによっては、これらのデフォルトフォルダがファイルシステムの書き込み可能な場所に存在しない場合があります。この場合、エラーが発生することがあります。
{{% /alert %}}

### ローカル artifact キャッシュのクリーンアップ

W&B はバージョン間で共通するファイルのダウンロードを高速化するため、artifact ファイルをキャッシュします。時間が経つと、このキャッシュディレクトリーが大きくなることがあります。キャッシュを整理し、最近使われていないファイルを削除するには [`wandb artifact cache cleanup`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-cache/" lang="ja" >}}) コマンドを実行してください。

以下のコードスニペットは、キャッシュサイズを 1GB に制限する方法を示しています。ターミナルにコピー＆ペーストしてお使いください。

```bash
$ wandb artifact cache cleanup 1GB
```