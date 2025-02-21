---
title: 'Tutorial: Create, track, and use a dataset artifact'
description: Artifacts クイックスタートでは、W&B を使用して、データセット アーティファクトを作成、追跡、および使用する方法を示します。
displayed_sidebar: default
menu:
  default:
    identifier: ja-guides-core-artifacts-artifacts-walkthrough
---

このウォークスルーでは、[W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) からデータセットアーティファクトを作成、トラッキング、使用する方法を示します。

## 1. W&B にログイン

W&B ライブラリをインポートして、W&B にログインします。まだ登録していない場合は、無料の W&B アカウントにサインアップする必要があります。

```python
import wandb

wandb.login()
```

## 2. Run を初期化

[`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) API を使ってバックグラウンドプロセスを生成し、W&B Run としてデータを同期およびログします。プロジェクト名とジョブタイプを指定します。

```python
# W&B Run を作成します。ここでは、データセットアーティファクトの
# 作成方法を示すために 'dataset' をジョブタイプとして指定しています。
run = wandb.init(project="artifacts-example", job_type="upload-dataset")
```

## 3. アーティファクトオブジェクトの作成

[`wandb.Artifact()`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) API を使ってアーティファクトオブジェクトを作成します。アーティファクトの `name` と `type` パラメータにファイルタイプの名前と説明を提供します。

たとえば、以下のコードスニペットは `‘bicycle-dataset’` というアーティファクトを `‘dataset’` のラベルで作成する方法を示しています。

```python
artifact = wandb.Artifact(name="bicycle-dataset", type="dataset")
```

アーティファクトの構築方法の詳細については、[Construct artifacts]({{< relref path="./construct-an-artifact.md" lang="ja" >}}) を参照してください。

## データセットをアーティファクトに追加

ファイルをアーティファクトに追加します。一般的なファイルタイプにはモデルやデータセットが含まれます。次の例では、マシン上にローカルで保存されている `dataset.h5` というデータセットをアーティファクトに追加しています。

```python
# アーティファクトの内容にファイルを追加します
artifact.add_file(local_path="dataset.h5")
```

前述のコードスニペットの `dataset.h5` ファイル名を、追加したいファイルのパスに置き換えてください。

## 4. データセットのログ

W&B の run オブジェクトの `log_artifact()` メソッドを使用して、アーティファクトバージョンを保存し、run の出力としてアーティファクトを宣言します。

```python
# アーティファクトバージョンを W&B に保存し、
# この run の出力としてマークします
run.log_artifact(artifact)
```

アーティファクトをログに記録すると、デフォルトで `'latest'` エイリアスが作成されます。アーティファクトのエイリアスとバージョンに関する詳細は、[Create a custom alias]({{< relref path="./create-a-custom-alias.md" lang="ja" >}}) と [Create new artifact versions]({{< relref path="./create-a-new-artifact-version.md" lang="ja" >}}) をそれぞれ参照してください。

## 5. アーティファクトのダウンロードと使用

次のコード例は、ログをとり、W&B サーバーに保存されたアーティファクトを使用するために取るべき手順を示しています。

1. まず、新しい run オブジェクトを **`wandb.init()`** で初期化します。
2. 次に、run オブジェクトの [`use_artifact()`]({{< relref path="/ref/python/run.md#use_artifact" lang="ja" >}}) メソッドを使用して、W&B に使用するアーティファクトを指定します。これにより、アーティファクトオブジェクトが返されます。
3. 3 番目に、アーティファクトの [`download()`]({{< relref path="/ref/python/artifact.md#download" lang="ja" >}}) メソッドを使用してアーティファクトの内容をダウンロードします。

```python
# W&B Run を作成します。ここでは、トレーニングをトラッキングするために
# run を使用するため 'type' に 'training' を指定しています。
run = wandb.init(project="artifacts-example", job_type="training")

# W&B にアーティファクトを問い合わせ、この run の入力としてマークします
artifact = run.use_artifact("bicycle-dataset:latest")

# アーティファクトの内容をダウンロードします
artifact_dir = artifact.download()
```

また、Public API (`wandb.Api`) を使用して、Run 外の W&B に既に保存されているデータをエクスポート（または更新）することもできます。詳細は [Track external files]({{< relref path="./track-external-files.md" lang="ja" >}}) を参照してください。