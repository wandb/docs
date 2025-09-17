---
title: Artifacts を作成する
description: W&B Artifact を作成・構築します。1 つ以上のファイルまたは URI 参照を Artifact に追加する方法を学びます。
menu:
  default:
    identifier: ja-guides-core-artifacts-construct-an-artifact
    parent: artifacts
weight: 2
---

W&B Python SDK を使用して [W&B Runs]({{< relref path="/ref/python/sdk/classes/run.md" lang="ja" >}}) からアーティファクトを構築します。[ファイル、ディレクトリー、URI、並列 Run からのファイル]({{< relref path="#add-files-to-an-artifact" lang="ja" >}}) をアーティファクトに追加できます。ファイルをアーティファクトに追加したら、アーティファクトを W&B Server または[独自のプライベートサーバー]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) に保存します。
Amazon S3 に保存されているファイルなどの外部ファイルの追跡方法については、[外部ファイルを追跡する]({{< relref path="./track-external-files.md" lang="ja" >}}) ページを参照してください。

## アーティファクトを構築する方法

[W&B Artifact]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) は 3 つのステップで構築します。

### 1. `wandb.Artifact()` でアーティファクト Python オブジェクトを作成する

[`wandb.Artifact()`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) クラスを初期化してアーティファクト オブジェクトを作成します。次のパラメータを指定します。

* **Name**: アーティファクトの名前を指定します。名前は一意で、記述的で、覚えやすいものにする必要があります。アーティファクトの名前は、W&B App の UI でアーティファクトを識別するためと、そのアーティファクトを使用したい場合の両方に使用します。
* **Type**: タイプを指定します。タイプはシンプルで記述的であり、機械学習パイプラインの単一のステップに対応している必要があります。一般的なアーティファクト タイプには、`'dataset'` または `'model'` があります。

{{% alert %}}
提供する「名前」と「タイプ」は、有向非巡回グラフの作成に使用されます。これにより、W&B App でアーティファクトのリネージを表示できます。
詳細については、[アーティファクト グラフを探索してトラバースする]({{< relref path="./explore-and-traverse-an-artifact-graph.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

{{% alert color="secondary" %}}
アーティファクトは同じ名前を持つことはできません。タイプパラメータに異なるタイプを指定した場合でも同様です。つまり、`dataset` タイプの `cats` という名前のアーティファクトと、同じ名前の `model` タイプの別のアーティファクトを作成することはできません。
{{% /alert %}}

アーティファクト オブジェクトを初期化する際に、オプションで説明とメタデータを提供できます。利用可能な属性とパラメータの詳細については、Python SDK リファレンスガイドの [`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) クラス定義を参照してください。
次の例は、データセット アーティファクトを作成する方法を示しています。

```python
import wandb

artifact = wandb.Artifact(name="<replace>", type="<replace>")
```

前のコードスニペットの文字列引数を、独自の名前とタイプに置き換えてください。

### 2. アーティファクトにファイルを 1 つ以上追加する

ファイル、ディレクトリー、外部 URI 参照 (Amazon S3 など) をアーティファクトのメソッドを使用して追加します。たとえば、単一のテキストファイルを追加するには、[`add_file`]({{< relref path="/ref/python/sdk/classes/artifact.md#add_file" lang="ja" >}}) メソッドを使用します。

```python
artifact.add_file(local_path="hello_world.txt", name="optional-name")
```

[`add_dir`]({{< relref path="/ref/python/sdk/classes/artifact.md#add_dir" lang="ja" >}}) メソッドを使用して複数のファイルを追加することもできます。ファイルを追加するには、[アーティファクトを更新する]({{< relref path="./update-an-artifact.md" lang="ja" >}}) を参照してください。

### 3. アーティファクトを W&B Server に保存する

最後に、アーティファクトを W&B Server に保存します。アーティファクトは Run に関連付けられています。したがって、Run オブジェクトの [`log_artifact()`]({{< relref path="/ref/python/sdk/classes/run.md#log_artifact" lang="ja" >}}) メソッドを使用してアーティファクトを保存します。

```python
# W&B Run を作成します。「job-type」を置き換えてください。
run = wandb.init(project="artifacts-example", job_type="job-type")

run.log_artifact(artifact)
```

{{% alert title="Artifact.save() と wandb.Run.log_artifact() の使い分け" %}}
- `Artifact.save()` を使用して、新しい Run を作成せずに既存のアーティファクトを更新します。
- `wandb.Run.log_artifact()` を使用して、新しいアーティファクトを作成し、特定の Run に関連付けます。
{{% /alert %}}

{{% alert color="secondary" %}}
`log_artifact` の呼び出しは、パフォーマンスの高いアップロードのために非同期に実行されます。これは、ループ内でアーティファクトをログに記録する際に予期せぬ振る舞いを引き起こす可能性があります。例:

```python
for i in range(10):
    a = wandb.Artifact(
        "race",
        type="dataset",
        metadata={
            "index": i,
        },
    )
    # ... アーティファクト a にファイルを追加 ...
    run.log_artifact(a)
```

アーティファクトは任意の順序でログに記録される可能性があるため、アーティファクト バージョン **v0** がメタデータ内でインデックス 0 を持つことは保証されません。
{{% /alert %}}

## アーティファクトにファイルを追加する

次のセクションでは、さまざまなファイルタイプや、並列 Run からアーティファクトを構築する方法を説明します。
次の例では、複数のファイルとディレクトリー構造を持つプロジェクト ディレクトリーがあると仮定します。

```
project-directory
|-- images
|   |-- cat.png
|   +-- dog.png
|-- checkpoints
|   +-- model.h5
+-- model.h5
```

### 単一ファイルを追加する

次のコードスニペットは、単一のローカルファイルをアーティファクトに追加する方法を示しています。

```python
# 単一ファイルを追加
artifact.add_file(local_path="path/file.format")
```

たとえば、作業中のローカルディレクトリーに `'file.txt'` という名前のファイルがあったとします。

```python
artifact.add_file("path/file.txt")  # `file.txt` として追加
```

アーティファクトには次のコンテンツが含まれています。

```
file.txt
```

オプションで、`name` パラメータにアーティファクト内の目的のパスを渡します。

```python
artifact.add_file(local_path="path/file.format", name="new/path/file.format")
```

アーティファクトは次のように保存されます。

```
new/path/file.txt
```

| API 呼び出し                                              | 結果のアーティファクト |
| --------------------------------------------------------- | ------------------ |
| `artifact.add_file('model.h5')`                           | model.h5           |
| `artifact.add_file('checkpoints/model.h5')`               | model.h5           |
| `artifact.add_file('model.h5', name='models/mymodel.h5')` | models/mymodel.h5  |

### 複数のファイルを追加する

次のコードスニペットは、ローカルディレクトリー全体をアーティファクトに追加する方法を示しています。

```python
# ディレクトリーを再帰的に追加
artifact.add_dir(local_path="path/file.format", name="optional-prefix")
```

次の API 呼び出しは、次のアーティファクト コンテンツを生成します。

| API 呼び出し                                | 結果のアーティファクト                                             |
| ------------------------------------------- | -------------------------------------------------------------- |
| `artifact.add_dir('images')`                | <p><code>cat.png</code></p><p><code>dog.png</code></p>         |
| `artifact.add_dir('images', name='images')` | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |
| `artifact.new_file('hello.txt')`            | `hello.txt`                                                    |

### URI 参照を追加する

URI が W&B ライブラリが処理できるスキームを持っている場合、アーティファクトは再現性のためにチェックサムやその他の情報を追跡します。
[`add_reference`]({{< relref path="/ref/python/sdk/classes/artifact.md#add_reference" lang="ja" >}}) メソッドを使用して、外部 URI 参照をアーティファクトに追加します。`'uri'` 文字列を独自の URI に置き換えてください。オプションで、name パラメータにアーティファクト内の目的のパスを渡します。

```python
# URI 参照を追加
artifact.add_reference(uri="uri", name="optional-name")
```

アーティファクトは現在、次の URI スキームをサポートしています。

* `http(s)://`: HTTP 経由でアクセス可能なファイルへのパス。HTTP サーバーが `ETag` および `Content-Length` 応答ヘッダーをサポートしている場合、アーティファクトは ETag とサイズメタデータの形式でチェックサムを追跡します。
* `s3://`: S3 内のオブジェクトまたはオブジェクト プレフィックスへのパス。アーティファクトは、参照されるオブジェクトのチェックサムとバージョン管理情報 (バケットでオブジェクト バージョン管理が有効になっている場合) を追跡します。オブジェクト プレフィックスは、プレフィックスの下のオブジェクトを最大 10,000 オブジェクトまで展開して含めます。
* `gs://`: GCS 内のオブジェクトまたはオブジェクト プレフィックスへのパス。アーティファクトは、参照されるオブジェクトのチェックサムとバージョン管理情報 (バケットでオブジェクト バージョン管理が有効になっている場合) を追跡します。オブジェクト プレフィックスは、プレフィックスの下のオブジェクトを最大 10,000 オブジェクトまで展開して含めます。

次の API 呼び出しは、次のアーティファクトを生成します。

| API 呼び出し                                                                      | 結果のアーティファクト コンテンツ                                      |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `artifact.add_reference('s3://my-bucket/model.h5')`                           | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/checkpoints/model.h5')`               | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/model.h5', name='models/mymodel.h5')` | `models/mymodel.h5`                                                  |
| `artifact.add_reference('s3://my-bucket/images')`                             | <p><code>cat.png</code></p><p><code>dog.png</code></p>               |
| `artifact.add_reference('s3://my-bucket/images', name='images')`              | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |

### 並列 Run からアーティファクトにファイルを追加する

大規模なデータセットや分散トレーニングの場合、複数の並列 Run が単一のアーティファクトに貢献する必要がある場合があります。

```python
import wandb
import time

# 並列で Run を開始するために Ray を使用します
# デモンストレーション目的のためです。
# 並列 Run は自由な方法でオーケストレーションできます。
import ray

ray.init()

artifact_type = "dataset"
artifact_name = "parallel-artifact"
table_name = "distributed_table"
parts_path = "parts"
num_parallel = 5

# 並列ライターの各バッチは、独自の
# 一意のグループ名を持つ必要があります。
group_name = "writer-group-{}".format(round(time.time()))


@ray.remote
def train(i):
    """
    ライター ジョブです。各ライターは 1 つの画像をアーティファクトに追加します。
    """
    with wandb.init(group=group_name) as run:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)

        # テーブルにデータを追加します。この例ではサンプルデータを使用します
        table = wandb.Table(columns=["a", "b", "c"], data=[[i, i * 2, 2**i]])

        # テーブルをアーティファクト内のフォルダーに追加します
        artifact.add(table, "{}/table_{}".format(parts_path, i))

        # アーティファクトを upsert すると、アーティファクトにデータが作成または追加されます
        run.upsert_artifact(artifact)


# 並列で Run を開始します
result_ids = [train.remote(i) for i in range(num_parallel)]

# すべてのライターがファイルを
# アーティファクトを完了する前に追加していることを確認するために結合します。
ray.get(result_ids)

# すべてのライターが完了したら、アーティファクトを完了して
# 準備完了としてマークします。
with wandb.init(group=group_name) as run:
    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    # テーブルのフォルダーを指す「パーティションテーブル」を作成し
    # それをアーティファクトに追加します。
    artifact.add(wandb.data_types.PartitionedTable(parts_path), table_name)

    # アーティファクトを完了すると、アーティファクトが最終決定され、このバージョンへの今後の upsert は許可されません。
    run.finish_artifact(artifact)
```