---
title: 'Tutorial: Create, track, and use a dataset artifact'
description: Artifacts クイックスタート では、W&B で データセット artifact を作成、追跡、使用する方法を紹介します。
displayed_sidebar: default
menu:
  default:
    identifier: ja-guides-core-artifacts-artifacts-walkthrough
---

このチュートリアルでは、[W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) からデータセット Artifactsを作成、追跡、および使用する方法を説明します。

## 1. W&B にログイン

W&B ライブラリをインポートし、W&B にログインします。まだお持ちでない場合は、無料の W&B アカウントにサインアップする必要があります。

```python
import wandb

wandb.login()
```

## 2. run の初期化

[`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) API を使用して、W&B Run としてデータを同期および記録するためのバックグラウンド プロセスを生成します。プロジェクト名とジョブタイプを指定します。

```python
# W&B Run を作成します。この例では、データセット Artifactsの作成方法を示すため、
# ジョブタイプとして「dataset」を指定します。
run = wandb.init(project="artifacts-example", job_type="upload-dataset")
```

## 3. artifact オブジェクトの作成

[`wandb.Artifact()`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) API を使用して、artifact オブジェクトを作成します。artifact の名前とファイルタイプの説明を、それぞれ `name` パラメータと `type` パラメータに指定します。

たとえば、次のコードスニペットは、`'dataset'` というラベルが付いた `'bicycle-dataset'` という artifact の作成方法を示しています。

```python
artifact = wandb.Artifact(name="bicycle-dataset", type="dataset")
```

artifact の構築方法の詳細については、[Artifacts の構築]({{< relref path="./construct-an-artifact.md" lang="ja" >}}) を参照してください。

## データセットを artifact に追加

artifact にファイルを追加します。一般的なファイルタイプには、モデルとデータセットが含まれます。次の例では、マシンにローカルに保存されている `dataset.h5` という名前のデータセットを artifact に追加します。

```python
# ファイルを artifact のコンテンツに追加します。
artifact.add_file(local_path="dataset.h5")
```

上記のコードスニペットのファイル名 `dataset.h5` を、artifact に追加するファイルへのパスに置き換えます。

## 4. データセットのログ

W&B run オブジェクトの `log_artifact()` メソッドを使用して、artifact の バージョンを保存し、artifact を run の出力として宣言します。

```python
# artifact のバージョンを W&B に保存し、この run の出力としてマークします。
run.log_artifact(artifact)
```

artifact をログに記録すると、デフォルトで `'latest'` エイリアスが作成されます。artifact のエイリアスとバージョンの詳細については、[カスタムエイリアスの作成]({{< relref path="./create-a-custom-alias.md" lang="ja" >}}) および [新しい artifact バージョンの作成]({{< relref path="./create-a-new-artifact-version.md" lang="ja" >}}) をそれぞれ参照してください。

## 5. artifact のダウンロードと使用

次のコード例は、W&B サーバーにログを記録して保存した artifact を使用するために実行できる手順を示しています。

1. まず、**`wandb.init()`** で新しい run オブジェクトを初期化します。
2. 次に、run オブジェクトの [`use_artifact()`]({{< relref path="/ref/python/run.md#use_artifact" lang="ja" >}}) メソッドを使用して、使用する artifact を W&B に指示します。これにより、artifact オブジェクトが返されます。
3. 3 番目に、artifact の [`download()`]({{< relref path="/ref/python/artifact.md#download" lang="ja" >}}) メソッドを使用して、artifact のコンテンツをダウンロードします。

```python
# W&B Run を作成します。ここでは、トレーニングを追跡するためにこの run を使用するため、
# 'type' に 'training' を指定します。
run = wandb.init(project="artifacts-example", job_type="training")

# W&B に artifact を照会し、この run への入力としてマークします。
artifact = run.use_artifact("bicycle-dataset:latest")

# artifact のコンテンツをダウンロードします。
artifact_dir = artifact.download()
```

または、Public API（`wandb.Api`）を使用して、Run の外部にある W&B にすでに保存されているデータをエクスポート（または更新）することもできます。詳細については、[外部ファイルの追跡]({{< relref path="./track-external-files.md" lang="ja" >}}) を参照してください。
