---
description: W&B アーティファクトを作成し、構築します。アーティファクトに1つ以上のファイルやURI参照を追加する方法を学びます。
displayed_sidebar: default
---


# Construct artifacts

<head>
  <title>Construct Artifacts</title>
</head>

W&B Python SDKを使って[W&B Runs](../../ref/python/run.md)からArtifactsを構築します。[ファイル、ディレクトリー、URI、および並列実行からのファイルをartefactsに追加](#add-files-to-an-artifact)できます。ファイルをartifactに追加した後、そのartifactをW&B Serverまたは[自分のプライベートサーバー](../hosting/hosting-options/self-managed.md)に保存します。

Amazon S3に保存されているファイルなどの外部ファイルを追跡する方法については、[外部ファイルを追跡する](./track-external-files.md)ページを参照してください。

## Artifactを構築する方法

[W&B Artifact](../../ref/python/artifact.md)を3つのステップで構築します：

### 1. `wandb.Artifact()`を使ってartifactのPythonオブジェクトを作成する

[`wandb.Artifact()`](../../ref/python/artifact.md)クラスを初期化してartifactオブジェクトを作成します。以下のパラメータを指定してください：

* **Name**: artifactの名前を指定します。その名前は唯一で、説明的で、覚えやすいものであるべきです。artifactの名前は、W&B App UIやartifactを使用するときに識別するために使用されます。
* **Type**: タイプを提供します。タイプはシンプルで説明的であり、機械学習パイプラインの特定のステップに対応するべきです。一般的なartifactのタイプとしては、`'dataset'`や`'model'`などがあります。

:::tip
提供した「name」と「type」は、有向非巡回グラフを作成するために使用されます。これにより、W&B Appでartifactのリネージを表示できます。

詳細については、[Explore and traverse artifact graphs](./explore-and-traverse-an-artifact-graph.md)を参照してください。
:::

:::caution
Artifactsは同じ名前を持つことはできません。たとえtypesパラメータに異なるタイプを指定した場合でも同様です。つまり、タイプが’dataset’の’cats’という名前のartifactや、同じ名前でタイプが’model’のartifactを作成することはできません。
:::

artifactオブジェクトを初期化する際に、オプションで説明やメタデータを提供できます。利用可能な属性やパラメータについての詳細は、Python SDKリファレンスガイドの[wandb.Artifact](../../ref/python/artifact.md)クラス定義を参照してください。

次の例は、dataset artifactを作成する方法を示しています：

```python
import wandb

artifact = wandb.Artifact(name="<replace>", type="<replace>")
```

前述のコードスニペットの文字列引数を自分の名前とタイプに置き換えてください。

### 2. artifactに1つ以上のファイルを追加する

artifactメソッドを使用してファイル、ディレクトリー、外部URI参照（Amazon S3など）などを追加できます。例えば、単一のテキストファイルを追加するには、[`add_file`](../../ref/python/artifact.md#add_file)メソッドを使用します：

```python
artifact.add_file(local_path="hello_world.txt", name="optional-name")
```

また、[`add_dir`](../../ref/python/artifact.md#add_dir)メソッドを使用して複数のファイルを追加することもできます。ファイルの追加方法の詳細については、[Update an artifact](./update-an-artifact.md)を参照してください。

### 3. artifactをW&Bサーバーに保存する

最後に、artifactをW&Bサーバーに保存します。Artifactsはrunに関連付けられています。そのため、runオブジェクトの[`log_artifact()`](../../ref/python/run#log\_artifact)メソッドを使用してartifactを保存します。

```python
# W&B Runを作成します。'job-type'を置き換えてください。
run = wandb.init(project="artifacts-example", job_type="job-type")

run.log_artifact(artifact)
```

オプションで、W&B runの外でartifactを構築することもできます。詳細については、[Track external files](./track-external-files)を参照してください。

:::caution
`log_artifact`の呼び出しは、パフォーマンスの向上のため非同期に行われます。これにより、ループ内でartifactをログ出力すると予期しない動作が発生することがあります。例えば：

```python
for i in range(10):
    a = wandb.Artifact(
        "race",
        type="dataset",
        metadata={
            "index": i,
        },
    )
    # ... artifact a にファイルを追加します ...
    run.log_artifact(a)
```

artifactバージョン**v0**は、そのメタデータのインデックスが0であることを保証しません。artifactは任意の順序でログに記録される可能性があります。
:::

## ファイルをartifactに追加する

次のセクションでは、異なるファイルタイプおよび並列runからartifactを構築する方法を示します。

次の例では、複数のファイルとディレクトリ構造を持つプロジェクトディレクトリーを想定します：

```
project-directory
|-- images
|   |-- cat.png
|   +-- dog.png
|-- checkpoints
|   +-- model.h5
+-- model.h5
```

### シングルファイルを追加する

次のコードスニペットは、単一のローカルファイルをartifactに追加する方法を示しています：

```python
# 単一のファイルを追加する
artifact.add_file(local_path="path/file.format")
```

例えば、作業ディレクトリーに`'file.txt'`という名前のファイルがあるとします。

```python
artifact.add_file("path/file.txt")  # `file.txt' として追加されます
```

artifactには次のコンテンツが含まれます：

```
file.txt
```

オプションで、artifact内の希望のパスを`name`パラメータに渡すことができます。

```python
artifact.add_file(local_path="path/file.format", name="new/path/file.format")
```

artifactは次のように保存されます：

```
new/path/file.txt
```

| API Call                                                  | 結果として得られるartifact |
| --------------------------------------------------------- | ------------------------- |
| `artifact.add_file('model.h5')`                           | model.h5                  |
| `artifact.add_file('checkpoints/model.h5')`               | model.h5                  |
| `artifact.add_file('model.h5', name='models/mymodel.h5')` | models/mymodel.h5         |

### 複数のファイルを追加する

次のコードスニペットは、ローカルディレクトリ全体を一括してartifactに追加する方法を示しています：

```python
# ディレクトリーを再帰的に追加する
artifact.add_dir(local_path="path/file.format", name="optional-prefix")
```

次のAPI呼び出しは、次のartifactコンテンツを生成します：

| API Call                                    | 結果として得られるartifact                                      |
| ------------------------------------------- | -------------------------------------------------------------- |
| `artifact.add_dir('images')`                | <p><code>cat.png</code></p><p><code>dog.png</code></p>         |
| `artifact.add_dir('images', name='images')` | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |
| `artifact.new_file('hello.txt')`            | `hello.txt`                                                    |

### URI参照を追加する

Artifactsは、URIがW&Bライブラリで処理できるスキームを持つ場合、再現性のためのチェックサムやその他の情報を追跡します。

[`add_reference`](../../ref/python/artifact#add\_reference)メソッドを使用して、外部URI参照をartifactに追加します。'uri'という文字列を自分のURIに置き換えてください。オプションで、希望のパスを`name`パラメータに渡すことができます。

```python
# URI参照を追加する
artifact.add_reference(uri="uri", name="optional-name")
```

artifactは現在、次のURIスキームをサポートしています：

* `http(s)://`: HTTPを介してアクセス可能なファイルへのパス。artifactは、HTTPサーバーが`ETag`および`Content-Length`レスポンスヘッダーをサポートしている場合、etagおよびサイズメタデータ形式のチェックサムを追跡します。
* `s3://`: S3内のオブジェクトまたはオブジェクトプレフィックスへのパス。artifactは、参照されたオブジェクトのチェックサムおよびバージョン管理情報（バケットがオブジェクトバージョン管理を有効にしている場合）を追跡します。オブジェクトプレフィックスは、最大10,000オブジェクトまでプレフィックス下のオブジェクトを含むように展開されます。
* `gs://`: GCS内のオブジェクトまたはオブジェクトプレフィックスへのパス。artifactは、参照されたオブジェクトのチェックサムおよびバージョン管理情報（バケットがオブジェクトバージョン管理を有効にしている場合）を追跡します。オブジェクトプレフィックスは、最大10,000オブジェクトまでプレフィックス下のオブジェクトを含むように展開されます。

次のAPI呼び出しは、次のArtifactsを生成します：

| API call                                                                      | 結果として得られるartifactコンテンツ                          |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `artifact.add_reference('s3://my-bucket/model.h5')`                           | `model.h5`                                                   |
| `artifact.add_reference('s3://my-bucket/checkpoints/model.h5')`               | `model.h5`                                                   |
| `artifact.add_reference('s3://my-bucket/model.h5', name='models/mymodel.h5')` | `models/mymodel.h5`                                          |
| `artifact.add_reference('s3://my-bucket/images')`                             | <p><code>cat.png</code></p><p><code>dog.png</code></p>       |
| `artifact.add_reference('s3://my-bucket/images', name='images')`              | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |

### 並列runからartifactにファイルを追加する

大規模なdatasetsや分散トレーニングでは、複数の並列runが単一のartifactに貢献する必要があります。

```python
import wandb
import time

# デモのために、rayを使用して並列runを開始します。好きなように並列runを調整できます。
import ray

ray.init()

artifact_type = "dataset"
artifact_name = "parallel-artifact"
table_name = "distributed_table"
parts_path = "parts"
num_parallel = 5

# 各バッチの並列ライターは、それぞれ独自のグループ名を持つべきです。
group_name = "writer-group-{}".format(round(time.time()))


@ray.remote
def train(i):
    """
    ライタージョブです。各ライターはartifactに1つの画像を追加します。
    """
    with wandb.init(group=group_name) as run:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)

        # データをwandbテーブルに追加します。この場合、例データを使用します。
        table = wandb.Table(columns=["a", "b", "c"], data=[[i, i * 2, 2**i]])

        # テーブルをartifact内のフォルダに追加します。
        artifact.add(table, "{}/table_{}".format(parts_path, i))

        # artifactを作成またはデータを追加します。
        run.upsert_artifact(artifact)


# 並列にrunを開始します。
result_ids = [train.remote(i) for i in range(num_parallel)]

# ライターのファイルが追加されるのを保証するためにすべてのライターに参加します。
ray.get(result_ids)

# すべてのライターが終了したら、artifactを終了して準備完了をマークします。
with wandb.init(group=group_name) as run:
    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    # フォルダ内のテーブルを指す"PartitioningTable"を作成して、artifactに追加します。
    artifact.add(wandb.data_types.PartitionedTable(parts_path), table_name)

    # artifactの終了は、このバージョンへの将来の"upserts"を禁止します。
    run.finish_artifact(artifact)
```