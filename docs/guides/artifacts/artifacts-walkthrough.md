---
description: Artifacts クイックスタートでは、W&B を使ってデータセットアーティファクトを作成、追跡、および使用する方法を示します。
displayed_sidebar: default
---


# Walkthrough

<head>
  <title>Walkthrough</title>
</head>

以下のウォークスルーでは、[W&B Runs](../runs/intro.md) からデータセットアーティファクトを作成、追跡、および使用するための主要なW&B Python SDKコマンドを示します。

## 1. W&Bにログイン

W&Bライブラリをインポートし、W&Bにログインします。まだW&Bの無料アカウントにサインアップしていない場合は、サインアップが必要です。

```python
import wandb

wandb.login()
```

## 2. Runを初期化

[`wandb.init()`](../../ref/python/init.md) APIを使用してバックグラウンドプロセスを生成し、データを同期およびログとしてW&B Runに記録します。プロジェクト名とジョブタイプを指定します。

```python
# W&B Runを作成します。この例ではデータセットアーティファクトの作成方法を示すため、
# ジョブタイプとして 'dataset' を指定しています。
run = wandb.init(project="artifacts-example", job_type="upload-dataset")
```

## 3. アーティファクトオブジェクトを作成

[`wandb.Artifact()`](../../ref/python/artifact.md) APIを使用してアーティファクトオブジェクトを作成します。アーティファクトの名前とファイルタイプの説明をそれぞれ `name` と `type` パラメータに提供します。

例えば、以下のコードスニペットは `‘bicycle-dataset’` という名前のアーティファクトを `‘dataset’` ラベルで作成する方法を示しています。

```python
artifact = wandb.Artifact(name="bicycle-dataset", type="dataset")
```

アーティファクトの構築方法について詳しくは、[Construct artifacts](./construct-an-artifact.md) を参照してください。

## データセットをアーティファクトに追加

アーティファクトにファイルを追加します。一般的なファイルタイプにはModelsやDatasetsが含まれます。次の例では、ローカルマシンに保存されている `dataset.h5` という名前のデータセットをアーティファクトに追加します。

```python
# アーティファクトの内容にファイルを追加する
artifact.add_file(local_path="dataset.h5")
```

上記のコードスニペット内の `dataset.h5` を追加したいファイルのパスに置き換えてください。

## 4. データセットをログ

W&B Runオブジェクトの `log_artifact()` メソッドを使用して、アーティファクトバージョンを保存し、Runの出力としてアーティファクトを宣言します。

```python
# アーティファクトバージョンをW&Bに保存し、
# このRunの出力としてマークする
run.log_artifact(artifact)
```

アーティファクトをログに記録すると、デフォルトで `'latest'` というエイリアスが作成されます。アーティファクトのエイリアスとバージョンについて詳しくは、[Create a custom alias](./create-a-custom-alias.md) および [Create new artifact versions](./create-a-new-artifact-version.md) を参照してください。

## 5. アーティファクトをダウンロードして使用

次のコード例では、ログに記録し保存したアーティファクトを使用する手順を示します。

1. まず、**`wandb.init()`** で新しいRunオブジェクトを初期化します。
2. 次に、Runオブジェクトの[`use_artifact()`](../../ref/python/run.md#use_artifact) メソッドを使用して、W&Bにどのアーティファクトを使用するかを指示します。これにより、アーティファクトオブジェクトが返されます。
3. 最後に、アーティファクトの[`download()`](../../ref/python/artifact.md#download) メソッドを使用して、アーティファクトの内容をダウンロードします。

```python
# W&B Runを作成します。ここでは 'type' として 'training' を指定しています。
# このRunはトレーニングの追跡に使用します。
run = wandb.init(project="artifacts-example", job_type="training")

# W&Bにアーティファクトを問い合わせ、このRunの入力としてマークする
artifact = run.use_artifact("bicycle-dataset:latest")

# アーティファクトの内容をダウンロードする
artifact_dir = artifact.download()
```

また、Public API (`wandb.Api`) を使用してRunの外部でW&Bに既に保存されているデータをエクスポート（または更新）することもできます。詳しくは、[Track external files](./track-external-files.md) を参照してください。

