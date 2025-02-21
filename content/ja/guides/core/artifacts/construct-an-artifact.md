---
title: Create an artifact
description: W\&B Artifact を作成、構築します。1 つまたは複数のファイル、または URI 参照を Artifact に追加する方法を学びます。
menu:
  default:
    identifier: ja-guides-core-artifacts-construct-an-artifact
    parent: artifacts
weight: 2
---

W&B Python SDK を使用して、[W&B Runs]({{< relref path="/ref/python/run.md" lang="ja" >}}) から Artifacts を構築します。[ファイル、ディレクトリー、URI、および並列 run からのファイルを Artifacts に追加]({{< relref path="#add-files-to-an-artifact" lang="ja" >}})できます。ファイルを Artifacts に追加したら、Artifacts を W&B サーバーまたは[独自のプライベートサーバー]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})に保存します。

Amazon S3 に保存されているファイルなど、外部ファイルを追跡する方法については、[外部ファイルを追跡]({{< relref path="./track-external-files.md" lang="ja" >}})のページを参照してください。

## Artifacts を構築する方法

[W&B Artifact]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) は、次の 3 つの手順で構築します。

### 1. `wandb.Artifact()` で Artifacts Python オブジェクトを作成する

[`wandb.Artifact()`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) クラスを初期化して、Artifacts オブジェクトを作成します。次のパラメーターを指定します。

*   **名前**: Artifacts の名前を指定します。名前は、一意で説明的で、覚えやすいものにする必要があります。Artifacts の名前を使用して、W&B App UI で Artifacts を識別したり、その Artifacts を使用する場合に使用したりします。
*   **タイプ**: タイプを指定します。タイプは、シンプルで説明的で、機械学習パイプラインの単一のステップに対応している必要があります。一般的な Artifacts タイプには、`'dataset'` または `'model'` があります。

{{% alert %}}
指定した「名前」と「タイプ」は、有向非巡回グラフの作成に使用されます。つまり、W&B App で Artifacts のリネージを表示できます。

詳細については、[Artifacts グラフの探索とトラバース]({{< relref path="./explore-and-traverse-an-artifact-graph.md" lang="ja" >}})を参照してください。
{{% /alert %}}

{{% alert color="secondary" %}}
types パラメータに異なるタイプを指定した場合でも、Artifacts は同じ名前を持つことはできません。つまり、タイプ `dataset` の `cats` という名前の Artifacts と、タイプ `model` の同じ名前の別の Artifacts を作成することはできません。
{{% /alert %}}

必要に応じて、Artifacts オブジェクトを初期化するときに、説明とメタデータを提供できます。使用可能な属性とパラメーターの詳細については、Python SDK リファレンスガイドの[`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}})クラス定義を参照してください。

次の例は、データセット Artifacts を作成する方法を示しています。

```python
import wandb

artifact = wandb.Artifact(name="<replace>", type="<replace>")
```

上記のコードスニペットの文字列引数を、独自の名前とタイプに置き換えます。

### 2. Artifacts に 1 つ以上のファイルを追加する

Artifacts メソッドを使用して、ファイル、ディレクトリー、外部 URI 参照 (Amazon S3 など) などを追加します。たとえば、単一のテキストファイルを追加するには、[`add_file`]({{< relref path="/ref/python/artifact.md#add_file" lang="ja" >}}) メソッドを使用します。

```python
artifact.add_file(local_path="hello_world.txt", name="optional-name")
```

[`add_dir`]({{< relref path="/ref/python/artifact.md#add_dir" lang="ja" >}}) メソッドを使用して、複数のファイルを追加することもできます。ファイルの追加方法の詳細については、[Artifacts を更新する]({{< relref path="./update-an-artifact.md" lang="ja" >}})を参照してください。

### 3. Artifacts を W&B サーバーに保存する

最後に、Artifacts を W&B サーバーに保存します。Artifacts は run に関連付けられています。したがって、run オブジェクトの [`log_artifact()`]({{< relref path="/ref/python/run.md#log_artifact" lang="ja" >}}) メソッドを使用して、Artifacts を保存します。

```python
# W&B Run を作成します。「job-type」を置き換えます。
run = wandb.init(project="artifacts-example", job_type="job-type")

run.log_artifact(artifact)
```

必要に応じて、W&B run の外部で Artifacts を構築できます。詳細については、[外部ファイルを追跡]({{< relref path="./track-external-files.md" lang="ja" >}})を参照してください。

{{% alert color="secondary" %}}
`log_artifact` の呼び出しは、パフォーマンスの高いアップロードのために非同期的に実行されます。これにより、ループで Artifacts をログに記録するときに、驚くような動作が発生する可能性があります。例:

```python
for i in range(10):
    a = wandb.Artifact(
        "race",
        type="dataset",
        metadata={
            "index": i,
        },
    )
    # ... artifact a にファイルを追加 ...
    run.log_artifact(a)
```

Artifacts バージョン **v0** は、Artifacts が任意の順序でログに記録される可能性があるため、メタデータに 0 のインデックスを持つことは保証されていません。
{{% /alert %}}

## Artifacts にファイルを追加する

次のセクションでは、さまざまなファイルタイプと並列 run から Artifacts を構築する方法について説明します。

次の例では、複数のファイルとディレクトリー構造を持つプロジェクトディレクトリーがあると仮定します。

```
project-directory
|-- images
|   |-- cat.png
|   +-- dog.png
|-- checkpoints
|   +-- model.h5
+-- model.h5
```

### 単一のファイルを追加する

次のコードスニペットは、単一のローカルファイルを Artifacts に追加する方法を示しています。

```python
# 単一のファイルを追加
artifact.add_file(local_path="path/file.format")
```

たとえば、作業ローカルディレクトリーに `'file.txt'` というファイルがあるとします。

```python
artifact.add_file("path/file.txt")  # `file.txt` として追加されました
```

Artifacts には次のコンテンツが含まれています。

```
file.txt
```

必要に応じて、`name` パラメーターに Artifacts 内の目的のパスを渡します。

```python
artifact.add_file(local_path="path/file.format", name="new/path/file.format")
```

Artifacts は次のように保存されます。

```
new/path/file.txt
```

| API 呼び出し                                                  | 結果の Artifacts |
| --------------------------------------------------------- | ------------------ |
| `artifact.add_file('model.h5')`                           | model.h5           |
| `artifact.add_file('checkpoints/model.h5')`               | model.h5           |
| `artifact.add_file('model.h5', name='models/mymodel.h5')` | models/mymodel.h5  |

### 複数のファイルを追加する

次のコードスニペットは、ローカルディレクトリー全体を Artifacts に追加する方法を示しています。

```python
# ディレクトリーを再帰的に追加
artifact.add_dir(local_path="path/file.format", name="optional-prefix")
```

上記の API 呼び出しにより、上記の Artifacts コンテンツが生成されます。

| API 呼び出し                                    | 結果の Artifacts                                     |
| ------------------------------------------- | ------------------------------------------------------ |
| `artifact.add_dir('images')`                | <p><code>cat.png</code></p><p><code>dog.png</code></p> |
| `artifact.add_dir('images', name='images')` | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |
| `artifact.new_file('hello.txt')`            | `hello.txt`                                            |

### URI 参照を追加する

Artifacts は、W&B ライブラリが処理方法を知っているスキームを URI が持っている場合、再現性のためにチェックサムやその他の情報を追跡します。

[`add_reference`]({{< relref path="/ref/python/artifact.md#add_reference" lang="ja" >}}) メソッドを使用して、外部 URI 参照を Artifacts に追加します。`'uri'` 文字列を独自の URI に置き換えます。必要に応じて、name パラメーターに Artifacts 内の目的のパスを渡します。

```python
# URI 参照を追加
artifact.add_reference(uri="uri", name="optional-name")
```

Artifacts は現在、次の URI スキームをサポートしています。

*   `http(s)://`: HTTP 経由でアクセスできるファイルへのパス。Artifacts は、HTTP サーバーが `ETag` および `Content-Length` 応答ヘッダーをサポートしている場合、etag およびサイズメタデータの形式でチェックサムを追跡します。
*   `s3://`: S3 内のオブジェクトまたはオブジェクトプレフィックスへのパス。Artifacts は、参照されているオブジェクトのチェックサムとバージョン管理情報 (バケットでオブジェクトのバージョン管理が有効になっている場合) を追跡します。オブジェクトプレフィックスは、プレフィックスの下にあるオブジェクトを含むように展開されます (最大 10,000 個のオブジェクト)。
*   `gs://`: GCS 内のオブジェクトまたはオブジェクトプレフィックスへのパス。Artifacts は、参照されているオブジェクトのチェックサムとバージョン管理情報 (バケットでオブジェクトのバージョン管理が有効になっている場合) を追跡します。オブジェクトプレフィックスは、プレフィックスの下にあるオブジェクトを含むように展開されます (最大 10,000 個のオブジェクト)。

上記の API 呼び出しは、上記の Artifacts を生成します。

| API 呼び出し                                                                      | 結果の Artifacts コンテンツ                                          |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `artifact.add_reference('s3://my-bucket/model.h5')`                           | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/checkpoints/model.h5')`               | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/model.h5', name='models/mymodel.h5')` | `models/mymodel.h5`                                                  |
| `artifact.add_reference('s3://my-bucket/images')`                             | <p><code>cat.png</code></p><p><code>dog.png</code></p>               |
| `artifact.add_reference('s3://my-bucket/images', name='images')`              | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |

### 並列 run から Artifacts にファイルを追加する

大規模なデータセットまたは分散トレーニングの場合、複数の並列 run が単一の Artifacts に貢献する必要がある場合があります。

```python
import wandb
import time

# デモンストレーションを目的として、ray を使用して並列で run を起動します。
# 並列 run は、好きなように調整できます。
import ray

ray.init()

artifact_type = "dataset"
artifact_name = "parallel-artifact"
table_name = "distributed_table"
parts_path = "parts"
num_parallel = 5

# 並列ライターの各バッチには、独自の
# 一意のグループ名が必要です。
group_name = "writer-group-{}".format(round(time.time()))


@ray.remote
def train(i):
    """
    ライタージョブ。各ライターは Artifacts に 1 つのイメージを追加します。
    """
    with wandb.init(group=group_name) as run:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)

        # データを wandb テーブルに追加します。この場合、サンプルデータを使用します
        table = wandb.Table(columns=["a", "b", "c"], data=[[i, i * 2, 2**i]])

        # テーブルを Artifacts 内のフォルダーに追加します
        artifact.add(table, "{}/table_{}".format(parts_path, i))

        # Artifacts の upsert は、Artifacts にデータを作成または追加します
        run.upsert_artifact(artifact)


# 並列で run を起動します
result_ids = [train.remote(i) for i in range(num_parallel)]

# すべてのライターに参加して、ファイルが
# Artifacts を終了する前にファイルが追加されていることを確認します。
ray.get(result_ids)

# すべてのライターが終了したら、Artifacts を終了します
# 準備ができていることをマークします。
with wandb.init(group=group_name) as run:
    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    # テーブルのフォルダーを指す「PartitionTable」を作成します
    # Artifacts に追加します。
    artifact.add(wandb.data_types.PartitionedTable(parts_path), table_name)

    # Artifacts の終了は Artifacts を確定し、将来の「upsert」を許可しません
    # このバージョンに。
    run.finish_artifact(artifact)
```