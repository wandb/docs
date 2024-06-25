---
description: W&B Artifacts のストレージやメモリ割り当てを管理します。
displayed_sidebar: default
---


# Storage

<head>
    <title>Artifact Storage</title>
</head>

W&Bは、アーティファクトファイルをデフォルトで米国に位置するプライベートなGoogle Cloud Storageバケットに保存します。すべてのファイルは保存時および転送時に暗号化されています。

機密性の高いファイルについては、[Private Hosting](../hosting/intro.md)を設定するか、[reference artifacts](./track-external-files.md)を使用することをお勧めします。

トレーニング中、W&Bはローカルでログ、アーティファクト、設定ファイルを以下のローカルディレクトリーに保存します:

| ファイル      | デフォルトの保存場所  | デフォルトの保存場所を変更するには:                                   |
| ------------- | --------------------- | -------------------------------------------------------------------- |
| logs          | `./wandb`             | `wandb.init` の `dir` または `WANDB_DIR` 環境変数を設定               |
| artifacts     | `~/.cache/wandb`      | `WANDB_CACHE_DIR` 環境変数を設定                                     |
| configs       | `~/.config/wandb`     | `WANDB_CONFIG_DIR` 環境変数を設定                                    |

:::caution
`wandb`が初期化されるマシンによっては、これらのデフォルトフォルダーがファイルシステムの書き込み可能な部分に存在しない場合があります。これがエラーを引き起こす可能性があります。
:::

### ローカルのアーティファクトキャッシュのクリーンアップ

W&Bは、共通のファイルを共有するバージョン間でダウンロード速度を上げるためにアーティファクトファイルをキャッシュします。時間が経つにつれて、このキャッシュディレクトリーは大きくなることがあります。キャッシュを整理し、最近使用されていないファイルを削除するには、[`wandb artifact cache cleanup`](../../ref/cli/wandb-artifact/wandb-artifact-cache/README.md)コマンドを実行します。

以下のコードスニペットは、キャッシュのサイズを1GBに制限する方法を示しています。ターミナルにコードスニペットをコピーして貼り付けてください:

```bash
$ wandb artifact cache cleanup 1GB
```

### 各アーティファクトバージョンはどれくらいのストレージを使用しますか？

2つのアーティファクトバージョン間で変更されたファイルのみがストレージコストを発生させます。

![v1 of the artifact "dataset" only has 2/5 images that differ, so it only uses 40% of the space.](@site/static/images/artifacts/artifacts-dedupe.PNG)

例えば、猫.pngと犬.pngの2つの画像ファイルを含む`animals`という名前のイメージアーティファクトを作成したとします:

```
images
|-- cat.png (2MB) # `v0` に追加
|-- dog.png (1MB) # `v0` に追加
```

このアーティファクトは自動的に`v0`というバージョンが割り当てられます。

もし新しい画像`rat.png`をアーティファクトに追加すると、新しいアーティファクトバージョン`v1`が作成され、以下の内容になります:

```
images
|-- cat.png (2MB) # `v0` に追加
|-- dog.png (1MB) # `v0` に追加
|-- rat.png (3MB) # `v1` に追加
```

`v1`は合計6MBのファイルを追跡しますが、共通部分の3MBを共有するため、実際には3MBのスペースしか使用しません。`v1`を削除すると、`rat.png`に関連する3MBのストレージが回収されます。`v0`を削除すると、`cat.png`と`dog.png`のストレージコストを引き継ぐため、`v1`のストレージサイズは6MBになります。