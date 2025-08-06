---
title: 'チュートリアル: データセットアーティファクトの作成・トラッキング・利用'
description: Artifacts クイックスタートでは、W&B でデータセット artifact を作成、追跡、利用する方法を紹介します。
displayed_sidebar: default
---

このウォークスルーでは、[W&B Runs]({{< relref "/guides/models/track/runs/" >}}) からデータセットアーティファクトを作成、トラッキング、利用する方法を解説します。

## 1. W&B にログインする

W&B ライブラリをインポートし、W&B にログインします。まだアカウントをお持ちでない場合は、無料の W&B アカウントにサインアップしてください。

```python
import wandb

wandb.login()
```

## 2. run を初期化する

[`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) API を使って、W&B Run としてデータを同期・記録するバックグラウンドプロセスを立ち上げます。プロジェクト名とジョブタイプを指定します。

```python
# W&B Run を作成します。ここでは 'dataset' をジョブタイプとして指定しています。
# この例ではデータセットアーティファクトの作成方法を示しています。
run = wandb.init(project="artifacts-example", job_type="upload-dataset")
```

## 3. アーティファクトオブジェクトを作成する

[`wandb.Artifact()`]({{< relref "/ref/python/sdk/classes/artifact.md" >}}) API を使用してアーティファクトオブジェクトを作ります。`name` パラメータにアーティファクトの名前を、`type` パラメータにファイルの種別を指定します。

例えば、以下のコードスニペットは `‘bicycle-dataset’` という名前のアーティファクトを `‘dataset’` タイプで作成しています。

```python
artifact = wandb.Artifact(name="bicycle-dataset", type="dataset")
```

アーティファクトの作成方法について詳しくは [Construct artifacts]({{< relref "./construct-an-artifact.md" >}}) をご覧ください。

## データセットをアーティファクトに追加する

アーティファクトにファイルを追加します。一般的なファイルタイプはモデルやデータセットです。以下はローカルマシン上に保存されている `dataset.h5` というデータセットファイルをアーティファクトに追加する例です。

```python
# アーティファクトの内容にファイルを追加する
artifact.add_file(local_path="dataset.h5")
```

上記のコードスニペット中のファイル名 `dataset.h5` を、追加したいファイルのパスに置き換えてください。

## 4. データセットをログに記録する

W&B run オブジェクトの `log_artifact()` メソッドを使って、アーティファクトのバージョンを保存し、その run の出力としてマークします。

```python
# アーティファクトのバージョンを W&B に保存し、
# この run の出力としてマークします
run.log_artifact(artifact)
```

アーティファクトをログに記録すると、デフォルトで `'latest'` というエイリアスが作成されます。エイリアスやバージョンの詳細については [Create a custom alias]({{< relref "./create-a-custom-alias.md" >}}) と [Create new artifact versions]({{< relref "./create-a-new-artifact-version.md" >}}) を参照してください。

## 5. アーティファクトをダウンロードして利用する

以下のコード例は、保存済みのアーティファクトを取得し、W&B サーバーから利用する流れを示しています。

1. まず、新しい run オブジェクトを **`wandb.init()`** で初期化します。
2. 次に、run オブジェクトの [`use_artifact()`]({{< relref "/ref/python/sdk/classes/run.md#use_artifact" >}}) メソッドでどのアーティファクトを使用するか指定します。これによりアーティファクトオブジェクトが返されます。
3. 最後に、artifacts の [`download()`]({{< relref "/ref/python/sdk/classes/artifact.md#download" >}}) メソッドでアーティファクトの内容をダウンロードします。

```python
# W&B Run を作成します。ここでは 'training' をジョブタイプに指定しています。
# この run でトレーニングを追跡するためです。
run = wandb.init(project="artifacts-example", job_type="training")

# W&B にアーティファクトを問い合わせ、この run の入力としてマークする
artifact = run.use_artifact("bicycle-dataset:latest")

# アーティファクトの内容をダウンロードする
artifact_dir = artifact.download()
```

また、Public API（`wandb.Api`）を使って、Run の外で既に保存された W&B 上のデータをエクスポート（または更新）することも可能です。詳しくは [Track external files]({{< relref "./track-external-files.md" >}}) をご覧ください。