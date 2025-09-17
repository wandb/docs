---
title: 'チュートリアル: Dataset Artifact の作成、追跡、使用'
description: Artifacts クイックスタートでは、W&B で Dataset Artifact を作成、追跡、使用する方法を紹介します。
displayed_sidebar: default
menu:
  default:
    identifier: ja-guides-core-artifacts-artifacts-walkthrough
---

このウォークスルーは、[W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) から Dataset Artifact を作成、追跡、および使用する方法を示します。

## 1. W&B にログインする

W&B ライブラリをインポートし、W&B にログインします。まだ無料の W&B アカウントにサインアップしていない場合は、サインアップする必要があります。

```python
import wandb

wandb.login()
```

## 2. Run を初期化する

[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) API を使用して、W&B Run としてデータを同期およびログするためのバックグラウンド プロセスを生成します。Project 名とジョブタイプを指定します。

```python
# W&B Run を作成します。この例は Dataset Artifact を作成する方法を示すため、
# ジョブタイプとして 'dataset' を指定します。
run = wandb.init(project="artifacts-example", job_type="upload-dataset")
```

## 3. Artifact オブジェクトを作成する

[`wandb.Artifact()`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) API を使用して、Artifact オブジェクトを作成します。`name` パラメータには Artifact の名前を、`type` パラメータにはファイルタイプの記述をそれぞれ指定します。

例えば、以下のコードスニペットは、`‘bicycle-dataset’` という名前で `‘dataset’` ラベルの Artifact を作成する方法を示しています。

```python
artifact = wandb.Artifact(name="bicycle-dataset", type="dataset")
```

Artifact の構築方法の詳細については、[Artifact の構築]({{< relref path="./construct-an-artifact.md" lang="ja" >}}) を参照してください。

## Dataset を Artifact に追加する

Artifact にファイルを追加します。一般的なファイルタイプには、Models や Datasets があります。以下の例では、ローカルマシンに保存されている `dataset.h5` という名前の Dataset を Artifact に追加します。

```python
# Artifact のコンテンツにファイルを追加します
artifact.add_file(local_path="dataset.h5")
```

上記のコードスニペットのファイル名 `dataset.h5` を、Artifact に追加したいファイルのパスに置き換えてください。

## 4. Dataset をログする

W&B Run オブジェクトの `log_artifact()` メソッドを使用して、Artifact のバージョンを保存し、その Artifact を Run の出力として宣言します。

```python
# Artifact のバージョンを W&B に保存し、
# この Run の出力としてマークします
run.log_artifact(artifact)
```

Artifact をログすると、デフォルトで `'latest'` エイリアスが作成されます。Artifact のエイリアスとバージョンに関する詳細については、[カスタム エイリアスの作成]({{< relref path="./create-a-custom-alias.md" lang="ja" >}}) および [新しい Artifact バージョンの作成]({{< relref path="./create-a-new-artifact-version.md" lang="ja" >}}) をそれぞれ参照してください。

## 5. Artifact をダウンロードして使用する

以下のコード例は、W&B サーバーにログして保存した Artifact を使用するための手順を示しています。

1. まず、**`wandb.init()`** で新しい Run オブジェクトを初期化します。
2. 次に、Run オブジェクトの [`use_artifact()`]({{< relref path="/ref/python/sdk/classes/run.md#use_artifact" lang="ja" >}}) メソッドを使用して、どの Artifact を使用するかを W&B に伝えます。これにより、Artifact オブジェクトが返されます。
3. 3番目に、Artifact の [`download()`]({{< relref path="/ref/python/sdk/classes/artifact.md#download" lang="ja" >}}) メソッドを使用して、Artifact のコンテンツをダウンロードします。

```python
# W&B Run を作成します。ここでは、'type' に 'training' を指定しています。
# この Run をトレーニングの追跡に使用するためです。
run = wandb.init(project="artifacts-example", job_type="training")

# W&B に Artifact を問い合わせ、この Run の入力としてマークします
artifact = run.use_artifact("bicycle-dataset:latest")

# Artifact のコンテンツをダウンロードします
artifact_dir = artifact.download()
```

あるいは、Public API (`wandb.Api`) を使用して、Run の外部で W&B に既に保存されているデータをエクスポート (または更新) できます。詳細については、[外部ファイルの追跡]({{< relref path="./track-external-files.md" lang="ja" >}}) を参照してください。