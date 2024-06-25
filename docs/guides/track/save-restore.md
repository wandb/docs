---
description: クラウドにファイルを保存し、後でローカルに復元する
displayed_sidebar: default
---


# Save & Restore Files

<head>
  <title>クラウドにファイルを保存および復元する</title>
</head>

このガイドでは最初に、`wandb.save` を使ってクラウドにファイルを保存する方法を示し、その後、`wandb.restore` でローカルに再作成する方法を示します。

## ファイルを保存する

時には、数値データやメディアのピースをログに記録する代わりに、モデルの重みや他のログソフトウェアの出力、さらにはソースコードのようなファイル全体を記録したいことがあります。

ファイルをrunに関連付け、W&Bにアップロードする方法は2つあります。

1. `wandb.save(filename)` を使用する。
2. wandb runディレクトリーにファイルを置くと、runの終了時にアップロードされます。

:::info
runを[再開](../runs/resuming.md)する場合、`wandb.restore(filename)` を呼び出すことでファイルを回復できます。
:::

ファイルが書き込まれている間に同期したい場合は、`wandb.save`でファイル名またはglobを指定できます。

### `wandb.save` の例

完全な動作例については[このレポート](https://app.wandb.ai/lavanyashukla/save\_and\_restore/reports/Saving-and-Restoring-Models-with-W%26B--Vmlldzo3MDQ3Mw)を参照してください。

```python
# 現在のディレクトリからモデルファイルを保存
wandb.save("model.h5")

# "ckpt" を含むすべてのファイルを保存
wandb.save("../logs/*ckpt*")

# "checkpoint" で始まるファイルを保存
wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))
```

:::info
W&Bのローカルrunディレクトリーはデフォルトでスクリプトの相対パス `./wandb` 内にあり、パスは `run-20171023_105053-3o4933r0` のようになります。ここで `20171023_105053` はタイムスタンプ、`3o4933r0` はrunのIDです。`WANDB_DIR` [環境変数](environment-variables.md) を設定するか、[`wandb.init`](./launch.md)の `dir` キーワード引数を絶対パスに設定することで、そのディレクトリ内にファイルが書き込まれます。
:::

### 保存ポリシーと相対パス

`wandb.save` は **policy** 引数を受け取り、デフォルトでは "**live**" に設定されています。使用可能なポリシーは次のとおりです：

* **live (デフォルト)** - このファイルを即座にwandbサーバーに同期し、変更があれば再同期する
* **now** - このファイルを即座にwandbサーバーに同期し、変更されても同期し続けない
* **end** - runが終了した時にのみファイルを同期する

**base\_path** 引数を `wandb.save` に指定することもできます。これにより、ディレクトリの階層を維持できます。例えば：

```python
wandb.save(path="./results/eval/*", base_path="./results", policy="now")
```

これにより、パターンに一致するすべてのファイルがルートではなく `eval` フォルダーに保存されます。

:::info
`wandb.save` が呼び出されると、指定されたパスに存在するすべてのファイルが一覧表示され、それらのファイルのシンボリックリンクがrunディレクトリー (`wandb.run.dir`) 内に作成されます。`wandb.save` を呼び出した後に同じパスに新しいファイルを作成しても、それらは同期されません。ファイルを直接 `wandb.run.dir` に書き込むか、新しいファイルが作成された時は必ず `wandb.save` を呼び出してください。
:::

### wandb run ディレクトリにファイルを保存する例

ファイル `model.h5` は `wandb.run.dir` に保存され、トレーニング終了時にアップロードされます。

```python
import wandb

wandb.init()

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[wandb.keras.WandbCallback()],
)
model.save(os.path.join(wandb.run.dir, "model.h5"))
```

こちらは公開されている例のページです。ファイルタブで, `model-best.h5` があることがわかります。これはKerasインテグレーションによってデフォルトで自動的に保存されますが、手動でチェックポイントを保存し、runに関連付けて保存することもできます。

[ライブ例を見る →](https://app.wandb.ai/wandb/neurips-demo/runs/206aacqo/files)

![](/images/experiments/example_saving_file_to_directory.png)

## ファイルを復元する

`wandb.restore(filename)` を呼び出すことで、ファイルがローカルrunディレクトリに復元されます。通常、`filename` は以前の実験runによって生成され、`wandb.save` でクラウドにアップロードされたファイルを指します。この呼び出しにより、ファイルのローカルコピーが作成され、読み取りのために開かれたローカルファイルストリームが返されます。

一般的なユースケース：

- 過去のrunで生成されたモデルアーキテクチャーや重みを復元する（より複雑なバージョン管理のユースケースについては、[Artifacts](../artifacts/intro.md)を参照してください）
- 障害が発生した場合に最後のチェックポイントからトレーニングを再開する（重要な詳細については[再開](../runs/resuming.md)のセクションを参照してください）

### `wandb.restore` の例

完全な動作例については[このレポート](https://app.wandb.ai/lavanyashukla/save\_and\_restore/reports/Saving-and-Restoring-Models-with-W%26B--Vmlldzo3MDQ3Mw)を参照してください。

```python
# "vanpelt" ユーザーの "my-project" 内の特定のrunからモデルファイルを復元
best_model = wandb.restore("model-best.h5", run_path="vanpelt/my-project/a1b2c3d")

# チェックポイントから重みファイルを復元
# (注意: run_path が提供されていない場合、再開が設定されている必要があります)
weights_file = wandb.restore("weights.h5")
# フレームワークがファイル名を期待している場合、返されたオブジェクトの "name" 属性を使用
# 例: Kerasの場合
my_predefined_model.load_weights(weights_file.name)
```

> `run_path` を指定しない場合、runの[再開](../runs/resuming.md)を設定する必要があります。トレーニング外でファイルをプログラムでアクセスしたい場合は、[Run API](../../ref/python/run.md)を使用してください。

## よくある質問

### ファイルを無視する方法

`wandb/settings` ファイルを編集して、`ignore_globs` にカンマ区切りの[globs](https://en.wikipedia.org/wiki/Glob\_\(programming\))リストを設定できます。また、`WANDB_IGNORE_GLOBS` [環境変数](./environment-variables.md)を設定することもできます。一般的なユースケースは、私たちが自動的に作成するgitパッチのアップロードを防ぐことです。例: `WANDB_IGNORE_GLOBS=*.patch`

### ファイルを保存するディレクトリを変更する

AWS S3やGoogle Cloud Storageにデフォルトでファイルを保存する場合、次のエラーが発生することがあります：`events.out.tfevents.1581193870.gpt-tpu-finetune-8jzqk-2033426287 is a cloud storage url, can't save file to wandb.`

TensorBoardイベントファイルや同期したい他のファイルの保存ディレクトリを変更するには、ファイルを `wandb.run.dir` に保存してクラウドに同期させてください。

### runの名前を取得するには？

スクリプト内からrunの名前を使用したい場合は、`wandb.run.name` を使用できます。例: "blissful-waterfall-2"

runの表示名にアクセスする前にsaveを呼び出す必要があります：

```
run = wandb.init(...)
run.save()
print(run.name)
```

### ローカルから保存されたすべてのファイルをプッシュするにはどうすればいいですか？

`wandb.init` の後にスクリプトの上部で一度 `wandb.save("*.pt")` を呼び出すと、そのパターンに一致するすべてのファイルが即座に `wandb.run.dir` に保存されます。

### 既にクラウドストレージに同期されたローカルファイルを削除できますか？

`wandb sync --clean` というコマンドを実行すると、既にクラウドストレージに同期されたローカルファイルを削除できます。使用方法の詳細については `wandb sync --help` を参照してください。

### コードの状態を復元する場合はどうすればいいですか？

[コマンドラインツール](../../ref/cli/README.md)の`restore`コマンドを使用して、指定されたrunを実行したときのコードの状態に戻ります。

```shell
# ブランチを作成し、run $RUN_ID が実行されたときの
# コードの状態を復元
wandb restore $RUN_ID
```

### `wandb` はどのようにしてコードの状態をキャプチャしますか？

スクリプトから `wandb.init` が呼び出されると、コードがgitリポジトリにある場合、最後のgitコミットへのリンクが保存されます。未コミットの変更やリモートと同期されていない変更がある場合に備えて、差分パッチも作成されます。