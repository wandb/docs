---
title: 'チュートリアル: データセット アーティファクトの作成、トラッキング、利用'
description: Artifacts クイックスタート では、W&B でデータセット artifact を作成、追跡、活用する方法を紹介します。
displayed_sidebar: default
menu:
  default:
    identifier: ja-guides-core-artifacts-artifacts-walkthrough
---

このウォークスルーでは、[W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) からデータセット artifact を作成、トラッキング、利用する方法を説明します。

## 1. W&B へログイン

W&B ライブラリをインポートして、W&B にログインします。まだ W&B アカウントを持っていない場合は、無料アカウントを作成してください。

```python
import wandb

wandb.login()
```

## 2. run の初期化

[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) API を使って、バックグラウンドでデータを同期・ログするためのプロセスを生成します。プロジェクト名とジョブタイプを指定しましょう。

```python
# W&B Run を作成します。ここではジョブタイプに 'dataset' を指定します。
# この例ではデータセット artifact の作成方法を扱っています。
run = wandb.init(project="artifacts-example", job_type="upload-dataset")
```

## 3. artifact オブジェクトの作成

[`wandb.Artifact()`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) API を使用してアーティファクトオブジェクトを作成します。`name` には artifact の名前、`type` にはファイルタイプの説明を指定してください。

例えば、以下のコードスニペットでは、`bicycle-dataset` という名前で `dataset` ラベルを持つ artifact を作成しています。

```python
artifact = wandb.Artifact(name="bicycle-dataset", type="dataset")
```

artifact の作成方法について詳しくは、[Construct artifacts]({{< relref path="./construct-an-artifact.md" lang="ja" >}}) をご覧ください。

## データセットを artifact に追加

artifact にファイルを追加します。一般的なファイルタイプには models や datasets があります。次の例では、ローカルに保存された `dataset.h5` というデータセットを artifact に追加しています。

```python
# artifact の内容にファイルを追加
artifact.add_file(local_path="dataset.h5")
```

上記のコードスニペットで、`dataset.h5` というファイル名は、追加したいファイルのパスに置き換えてください。

## 4. データセットのログ

W&B の run オブジェクトの `log_artifact()` メソッドを使って、artifact バージョンを保存し、この run の出力として宣言します。

```python
# artifact バージョンを W&B に保存し、
# この run の出力としてマークします
run.log_artifact(artifact)
```

artifact をログすると、デフォルトで `'latest'` エイリアスが作成されます。artifact のエイリアスやバージョンについて詳しくは、[Create a custom alias]({{< relref path="./create-a-custom-alias.md" lang="ja" >}}) および [Create new artifact versions]({{< relref path="./create-a-new-artifact-version.md" lang="ja" >}}) をご覧ください。

## 5. artifact のダウンロードと利用

以下のコード例は、保存された artifact を W&B サーバーから取得して利用する手順を示しています。

1. まず、**`wandb.init()`** で新しい run オブジェクトを初期化します。
2. 次に、run オブジェクトの [`use_artifact()`]({{< relref path="/ref/python/sdk/classes/run.md#use_artifact" lang="ja" >}}) メソッドを使い、利用する artifact を指定します。このメソッドは artifact オブジェクトを返します。
3. 最後に、artifacts の [`download()`]({{< relref path="/ref/python/sdk/classes/artifact.md#download" lang="ja" >}}) メソッドで artifact の内容をダウンロードします。

```python
# W&B Run を作成します。ここでは 'type' に 'training' を指定しています。
# この run でトレーニングをトラッキングします。
run = wandb.init(project="artifacts-example", job_type="training")

# W&B から artifact を取得し、この run の入力として指定
artifact = run.use_artifact("bicycle-dataset:latest")

# artifact の内容をダウンロード
artifact_dir = artifact.download()
```

また、Public API (`wandb.Api`) を利用して、Run の外で W&B に保存済みデータのエクスポート（もしくはデータの更新）も可能です。詳しくは [Track external files]({{< relref path="./track-external-files.md" lang="ja" >}}) をご覧ください。