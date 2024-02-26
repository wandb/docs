---
description: Save files to the cloud and restore them locally later
displayed_sidebar: default
---

# ファイルの保存と復元

<head>
  <title>クラウドにファイルを保存し、復元する</title>
</head>

このガイドではまず、`wandb.save`を使ってクラウドにファイルを保存する方法を紹介し、次に`wandb.restore`を使って、それらをローカルに再作成する方法を紹介します。

## ファイルの保存

場合によっては、数値やメディアの一部を記録するのではなく、ファイル全体を記録したいことがあります。例えば、モデルの重みや、他のログソフトウェアの出力、ソースコードなどです。

W&Bにファイルを関連付けてアップロードする方法は2つあります。

1. `wandb.save(filename)` を使う。
2. wandb runディレクトリーにファイルを置くと、runの終了時にアップロードされます。

:::info
[再開](../runs/resuming.md)するrunであれば、`wandb.restore(filename)`を呼び出すことでファイルを復元できます。
:::

書き込まれているファイルを同期する場合は、`wandb.save`でファイル名やglobを指定できます。

### `wandb.save`の例

完全な動作例は、[こちらのレポート](https://app.wandb.ai/lavanyashukla/save\_and\_restore/reports/Saving-and-Restoring-Models-with-W%26B--Vmlldzo3MDQ3Mw)をご覧ください。

```python
# 現在のディレクトリからモデルファイルを保存する
wandb.save('model.h5')

# サブストリング "ckpt" を含むすべてのファイルを保存する
wandb.save('../logs/*ckpt*')

# "checkpoint"で始まるファイルを書き込み時に保存する
wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))
```

:::info
W&Bのローカルランディレクトリはデフォルトでスクリプトに関連する`./wandb`ディレクトリ内にありますが、パスは`run-20171023_105053-3o4933r0`のようになります。ここで`20171023_105053`はタイムスタンプで`3o4933r0`はランのIDです。`WANDB_DIR`[環境変数](environment-variables.md)を設定するか、[`wandb.init`](./launch.md)の`dir`キーワード引数に絶対パスを設定することで、代わりにそのディレクトリ内にファイルが書き込まれるようになります。
:::

### 保存ポリシーと相対パス

`wandb.save`は**policy**引数も受け付け、デフォルトで"**live**"に設定されています。利用可能なポリシーは以下の通りです。

* **live (デフォルト)** - このファイルをすぐにwandbサーバーに同期し、変更があれば再同期する
* **now** - このファイルをすぐにwandbサーバーに同期し、変更があっても再同期しない
* **end** - ランが終了したときにのみファイルを同期する

また、`wandb.save`に**base\_path**引数を指定することもできます。これによりディレクトリ階層を維持することができます。例えば：

```python
wandb.save(
    path="./results/eval/*", 
    base_path="./results", 
    policy="now"
    )    
```
すべてのファイルがパターンに一致すると、ルートの代わりに`eval`フォルダに保存されます。

:::info
`wandb.save`が呼び出されると、指定されたパスに存在するすべてのファイルを一覧表示し、runディレクトリー（`wandb.run.dir`）にシンボリックリンクを作成します。`wandb.save`を呼び出した後に同じパスに新しいファイルを作成する場合、それらのファイルは同期されません。ファイルは直接`wandb.run.dir`に書き込むか、新しいファイルが作成されるたびに`wandb.save`を呼び出す必要があります。
:::

### wandbの実行ディレクトリにファイルを保存する例

ファイル`model.h5`は`wandb.run.dir`に保存され、トレーニングの終了時にアップロードされます。

```python
import wandb
wandb.init()

model.fit(X_train, y_train,  validation_data=(X_test, y_test),
    callbacks=[wandb.keras.WandbCallback()])
model.save(os.path.join(wandb.run.dir, "model.h5"))
```

こちらが公開されている例のページです。ファイルタブで`model-best.h5`が表示されています。これはKerasの統合によってデフォルトで自動的に保存されますが、チェックポイントを手動で保存することもでき、runに関連付けて保存します。

[ライブ例を見る →](https://app.wandb.ai/wandb/neurips-demo/runs/206aacqo/files)

![](/images/experiments/example_saving_file_to_directory.png)

## ファイルの復元

`wandb.restore(filename)` を呼び出すと、ローカルのrunディレクトリーにファイルが復元されます。通常、`filename`は、以前の実験runで生成され、`wandb.save`を使用してクラウドにアップロードされたファイルを指します。この呼び出しにより、ファイルのローカルコピーが作成され、読み取り用に開かれたローカルファイルストリームが返されます。

一般的なユースケース:
過去のrunsで生成されたモデルアーキテクチャや重みを復元する（より複雑なバージョン管理のユースケースについては、[Artifacts](../artifacts/intro.md)を参照してください）。
* エラーが発生した場合、最後のチェックポイントからトレーニングを再開する（[再開](../runs/resuming.md)のセクションを参照してください）。

### `wandb.restore` の例

完全な動作例については、[こちらのレポート](https://app.wandb.ai/lavanyashukla/save\_and\_restore/reports/Saving-and-Restoring-Models-with-W%26B--Vmlldzo3MDQ3Mw) をご覧ください。

```python
# ユーザー "vanpelt" が "my-project" で実行した特定の run からモデルファイルを復元する
best_model = wandb.restore(
  'model-best.h5', run_path="vanpelt/my-project/a1b2c3d")

# チェックポイントから重みファイルを復元する
# (注意：run_pathが提供されていない場合、再開が設定されている必要があります)
weights_file = wandb.restore('weights.h5')
# フレームワークがファイル名を期待している場合、
# 返されたオブジェクトの "name" 属性を使ってください。
# 例：Keras では
my_predefined_model.load_weights(weights_file.name)
```

> `run_path`を指定しない場合、run の[再開](../runs/resuming.md)を設定する必要があります。トレーニング外でファイルにプログラムでアクセスしたい場合は、[Run API](../../ref/python/run.md)を使用してください。

## 一般的な質問

### どのようにしてファイルを無視するのか？

`wandb/settings` ファイルを編集して、`ignore_globs` を [globs](https://en.wikipedia.org/wiki/Glob\_\(programming\)) 形式のカンマ区切りのリストに設定することができます。また、`WANDB_IGNORE_GLOBS` [環境変数](./environment-variables.md)を設定することもできます。一般的なユースケースとしては、自動的に作成される git パッチのアップロードを防ぐために、`WANDB_IGNORE_GLOBS=*.patch` と設定します。

### ランが終了する前にファイルを同期する方法は？
長いrunがある場合は、runの終了前に、モデルのチェックポイントなどのファイルをクラウドにアップロードしたいことがあるでしょう。デフォルトでは、ほとんどのファイルのアップロードはrunの終了まで待っています。スクリプトに`wandb.save('*.pth')`や`wandb.save('latest.pth')`を追加することで、ファイルが作成されたり更新されたりするたびに上記のファイルをアップロードできます。

### ファイルを保存するディレクトリを変更する

AWS S3やGoogle Cloud Storageにデフォルトでファイルを保存する場合、以下のエラーが発生することがあります：`events.out.tfevents.1581193870.gpt-tpu-finetune-8jzqk-2033426287 is a cloud storage url, can't save file to wandb.`

TensorBoardのイベントファイルや同期させたい他のファイルのログディレクトリを変更するには、ファイルを`wandb.run.dir`に保存して、クラウドに同期させるようにします。

### ランの名前を取得する方法は？

スクリプト内でランの名前を使用したい場合は、`wandb.run.name`を使用して、例えば "blissful-waterfall-2" のようなランの名前を取得できます。

表示名にアクセスする前に、runでsaveを呼び出す必要があります。

```
run = wandb.init(...)
run.save()
print(run.name)
```

### ローカルからすべての保存されたファイルをプッシュする方法は？

`wandb.init`の後にスクリプトの先頭で一度`wandb.save("*.pt")`を呼び出すと、そのパターンに一致するすべてのファイルが、`wandb.run.dir`に書き込まれるとすぐに保存されます。

### すでにクラウドストレージに同期されたローカルファイルを削除することはできますか？

ローカルのファイルをすでにクラウドストレージに同期した後に削除するためのコマンド`wandb sync --clean`があります。使用方法の詳細は `wandb sync --help`で見ることができます。

### コードの状態を復元したい場合はどうすればいいですか？
`restore`コマンドを使って、特定のrunが実行された時のコードの状態に戻ります。[コマンドラインツール](../../ref/cli/README.md)を参照してください。

```python
# ブランチを作成し、コードの状態を復元します
# run $RUN_IDが実行された時の状態に
wandb restore $RUN_ID
```

### `wandb`はコードの状態をどのようにキャプチャしますか？

スクリプトから`wandb.init`が呼び出されると、コードがgitリポジトリにある場合は最後のgitコミットへのリンクが保存されます。また、未コミットの変更やリモートと同期されていない変更がある場合は、diffパッチも作成されます。