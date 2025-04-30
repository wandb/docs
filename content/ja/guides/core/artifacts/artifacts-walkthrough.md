---
title: 'チュートリアル: データセットアーティファクトを作成、追跡、利用する方法'
description: アーティファクト クイックスタートでは、W&B でデータセット アーティファクトを作成、追跡、使用する方法を示します。
displayed_sidebar: default
menu:
  default:
    identifier: ja-guides-core-artifacts-artifacts-walkthrough
---

このウォークスルーでは、[W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}})からデータセットアーティファクトを作成、追跡、使用する方法を示します。

## 1. W&B にログイン

W&B ライブラリをインポートし、W&B にログインします。まだアカウントをお持ちでない場合は、無料の W&B アカウントにサインアップする必要があります。

```python
import wandb

wandb.login()
```

## 2. Run を初期化

[`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) API を使用して、バックグラウンドプロセスを生成し、データを W&B Run として同期してログします。プロジェクト名とジョブタイプを指定してください。

```python
# W&B Run を作成します。この例ではデータセットアーティファクトを作成する方法を示しているので、
# ジョブタイプとして 'dataset' を指定します。
run = wandb.init(project="artifacts-example", job_type="upload-dataset")
```

## 3. アーティファクト オブジェクトを作成

[`wandb.Artifact()`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) API を使用してアーティファクトオブジェクトを作成します。アーティファクトの名前とファイルタイプの説明を、それぞれ `name` と `type` パラメータとして指定します。

例えば、次のコードスニペットは `‘bicycle-dataset’` という名前で、`‘dataset’` というラベルのアーティファクトを作成する方法を示しています。

```python
artifact = wandb.Artifact(name="bicycle-dataset", type="dataset")
```

アーティファクトの構築方法の詳細については、[Construct artifacts]({{< relref path="./construct-an-artifact.md" lang="ja" >}})を参照してください。

## データセットをアーティファクトに追加

ファイルをアーティファクトに追加します。一般的なファイルタイプには Models や Datasets が含まれます。次の例では、ローカルマシンに保存されている `dataset.h5` というデータセットをアーティファクトに追加します。

```python
# ファイルをアーティファクトのコンテンツに追加
artifact.add_file(local_path="dataset.h5")
```

上記のコードスニペットのファイル名 `dataset.h5` は、アーティファクトに追加したいファイルのパスに置き換えてください。

## 4. データセットをログ

W&B Run オブジェクトの `log_artifact()` メソッドを使用して、アーティファクトバージョンを保存し、アーティファクトを run の出力として宣言します。

```python
# アーティファクトバージョンを W&B に保存し、
# この run の出力としてマークします
run.log_artifact(artifact)
```

アーティファクトをログする際、'latest' エイリアスがデフォルトで作成されます。アーティファクトのエイリアスとバージョンに関する詳細については、[Create a custom alias]({{< relref path="./create-a-custom-alias.md" lang="ja" >}}) および [Create new artifact versions]({{< relref path="./create-a-new-artifact-version.md" lang="ja" >}}) をそれぞれ参照してください。

## 5. アーティファクトをダウンロードして使用

次のコード例では、ログされ保存されたアーティファクトを W&B サーバーで使用する手順を示します。

1. まず、**`wandb.init()`.** を使用して新しい run オブジェクトを初期化します。
2. 次に、run オブジェクトの [`use_artifact()`]({{< relref path="/ref/python/run.md#use_artifact" lang="ja" >}}) メソッドを使用して、W&B に使用するアーティファクトを指定します。これにより、アーティファクトオブジェクトが返されます。
3. 最後に、アーティファクトの [`download()`]({{< relref path="/ref/python/artifact.md#download" lang="ja" >}}) メソッドを使用してアーティファクトの内容をダウンロードします。

```python
# W&B Run を作成します。ここでは 'training' を 'type' として指定します
# この run をトレーニングの追跡に使用するためです。
run = wandb.init(project="artifacts-example", job_type="training")

# W&B にアーティファクトを検索させ、この run の入力としてマークします
artifact = run.use_artifact("bicycle-dataset:latest")

# アーティファクトの内容をダウンロードします
artifact_dir = artifact.download()
```

または、Public API (`wandb.Api`) を使用して、Run の外で W&B に既に保存されたデータをエクスポート（または更新）できます。詳細は [Track external files]({{< relref path="./track-external-files.md" lang="ja" >}}) を参照してください。