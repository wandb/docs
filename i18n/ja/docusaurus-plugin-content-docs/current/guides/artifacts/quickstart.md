---
description: >-
  Artifacts quickstart shows how to create, track, and use a dataset artifact
  with W&B.
displayed_sidebar: ja
---

# クイックスタート

<head>
  <title>Artifacts クイックスタート</title>
</head>

このクイックスタートでは、データセットのアーティファクトを作成、トラッキング、および使用する方法を示します。Weights & Biasesのアカウントをお持ちでない方は、始める前にお手続きください。

以下の手順は、アーティファクトの作成と使用方法を説明しています。 ステップ1と2は、W&B Artifactsに限らず一般的なものです。

1. [Weights & Biasesにログインします。](#log-into-weights--biasess)
2. [Runを初期化します。](#initialize-a-run)
3. [アーティファクトオブジェクトを作成します。](#create-an-artifact-object)
4. [データセットをアーティファクトに追加します。](#add-the-dataset-to-the-artifact)
5. [データセットをログに記録します。](#log-the-dataset)
6. [アーティファクトをダウンロードして使用します。](#download-and-use-the-artifact)
### Weights & Biasesにログイン

Weights & Biasesライブラリをインポートし、W&Bにログインします。まだ登録していない場合は、無料のW&Bアカウントに登録する必要があります。

```python
import wandb

wandb.login()
```

### runの初期化

[`wandb.init()`](https://docs.wandb.ai/ref/python/init) APIを使って、バックグラウンドプロセスを生成し、W&B Runとしてデータを同期およびログに記録できます。プロジェクト名とジョブタイプを指定してください：

```python
# W&B Runを作成します。ここでは、データセットアーティファクトの作成方法を示すため、
# ジョブタイプとして'dataset'を指定しています。
run = wandb.init(
    project="artifacts-example", 
    job_type='upload-dataset'
    )
```

### アーティファクトオブジェクトの作成

[`wandb.Artifact()`](https://docs.wandb.ai/ref/python/artifact) APIを使ってアーティファクトオブジェクトを作成します。アーティファクトに名前を付け、ファイルタイプの説明を`name`および`type`パラメータにそれぞれ指定してください。

例えば、以下のコードスニペットは、`‘dataset’`ラベルの`‘bicycle-dataset’`というアーティファクトを作成する方法を示しています：

```python
artifact = wandb.Artifact(
    name='bicycle-dataset', 
    type='dataset'
    )    
```
アーティファクトの構築方法についての詳細は、[アーティファクトの構築](https://docs.wandb.ai/guides/artifacts/construct-an-artifact)を参照してください。

### データセットをアーティファクトに追加

アーティファクトにファイルを追加します。一般的なファイルタイプには、モデルやデータセットがあります。以下の例では、ローカルマシンに保存されている`dataset.h5`という名前のデータセットをアーティファクトに追加しています。

```python
# アーティファクトの内容にファイルを追加
artifact.add_file(local_path='dataset.h5')
```

上記のコードスニペットの`dataset.h5`というファイル名を、アーティファクトに追加したいファイルへのパスに置き換えてください。

### データセットをログする
W&Bのrunオブジェクトの`log_artifact()`メソッドを使って、アーティファクトのバージョンを保存し、そのアーティファクトをrunの出力として宣言します。

```python
# アーティファクトのバージョンをW&Bに保存し、
# このrunの出力としてマークする
run.log_artifact(artifact)
```

アーティファクトをログすると、デフォルトで`'latest'`エイリアスが作成されます。アーティファクトのエイリアスとバージョンについての詳細は、それぞれ[カスタムエイリアスの作成](https://docs.wandb.ai/guides/artifacts/create-a-custom-alias)と[新しいアーティファクトバージョンの作成](https://docs.wandb.ai/guides/artifacts/create-a-new-artifact-version)をご覧ください。

### アーティファクトのダウンロードと使用

以下のコード例は、ログしたアーティファクトをWeights & Biasesサーバーに保存し、それを使用するための手順を示しています。

1. 最初に、**`wandb.init()`**で新しいrunオブジェクトを初期化します。
2. 次に、runオブジェクトの[`use_artifact()`](https://docs.wandb.ai/ref/python/run#use\_artifact)メソッドを使って、Weights & Biasesにどのアーティファクトを使用するか指示します。これによってアーティファクトオブジェクトが返されます。
3. 最後に、アーティファクトの[`download()`](https://docs.wandb.ai/ref/python/artifact#download)メソッドを使って、アーティファクトの内容をダウンロードします。
```python
# W&B Runを作成します。ここでは'type'に'training'を指定しています
# なぜなら、このrunでトレーニングのトラッキングを行うためです。
run = wandb.init(
    project="artifacts-example", 
    job_type='training'
    )

# アーティファクトをW&Bから取得し、このrunの入力としてマークします
artifact = run.use_artifact('bicycle-dataset:latest')

# アーティファクトの内容をダウンロードします
artifact_dir = artifact.download()
```
代わりに、公開API（`wandb.Api`）を使用して、Weights & Biases外部のRun以外ですでに保存されているデータをエクスポート（または更新）することもできます。詳細については、[外部ファイルのトラッキング](https://docs.wandb.ai/guides/artifacts/track-external-files)を参照してください。