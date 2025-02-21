---
title: Create an artifact
description: W&B アーティファクトを作成、構築します。アーティファクトに1つ以上のファイルまたは URI 参照を追加する方法を学びましょう。
menu:
  default:
    identifier: ja-guides-core-artifacts-construct-an-artifact
    parent: artifacts
weight: 2
---

W&B Python SDK を使用して、[W&B Runs]({{< relref path="/ref/python/run.md" lang="ja" >}}) からアーティファクトを構築します。[ファイル、ディレクトリ、URI、並列 run のファイルをアーティファクトに追加]({{< relref path="#add-files-to-an-artifact" lang="ja" >}})できます。ファイルをアーティファクトに追加した後、アーティファクトを W&B サーバーまたは[独自のプライベートサーバー]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})に保存します。

Amazon S3 に保存されたファイルのような外部ファイルを追跡する方法については、[外部ファイルを追跡する]({{< relref path="./track-external-files.md" lang="ja" >}})ページを参照してください。

## アーティファクトを構築する方法

3 つのステップで [W&B Artifact]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) を構築します：

### 1. `wandb.Artifact()` を使用してアーティファクトの Python オブジェクトを作成します

[`wandb.Artifact()`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) クラスを初期化してアーティファクトオブジェクトを作成します。以下のパラメータを指定してください：

* **Name**: アーティファクトの名前を指定します。名前は一意で、わかりやすく、記憶しやすいものである必要があります。アーティファクトの名前は、W&B アプリの UI 内でアーティファクトを識別したり、そのアーティファクトを使用したいときに役立ちます。
* **Type**: タイプを指定します。タイプはシンプルかつ説明的で、機械学習パイプラインの単一ステップに対応するものである必要があります。一般的なアーティファクトのタイプには `'dataset'` や `'model'` があります。

{{% alert %}}
提供された「名前」と「タイプ」は、有向非巡回グラフを作成するために使用されます。これにより、W&B アプリでアーティファクトのリネージを見ることができます。

詳細については、[アーティファクトグラフを探索しトラバースする]({{< relref path="./explore-and-traverse-an-artifact-graph.md" lang="ja" >}})を参照してください。
{{% /alert %}}

{{% alert color="secondary" %}}
アーティファクトは同じ名前を持つことはできません。たとえタイプパラメータに異なるタイプを指定してもです。言い換えれば、`cats` という名前のアーティファクトをタイプ `dataset` として作成し、同じ名前のアーティファクトをタイプ `model` として作成することはできません。
{{% /alert %}}

オプションとして、アーティファクトオブジェクトを初期化する際に、説明やメタデータを提供することができます。利用可能な属性とパラメータに関する詳細情報については、Python SDK リファレンスガイドの[`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}})クラスの定義を参照してください。

以下の例は、データセットアーティファクトを作成する方法を示しています：

```python
import wandb

artifact = wandb.Artifact(name="<replace>", type="<replace>")
```

上記のコードスニペットの文字列引数を独自の名前とタイプに置き換えてください。

### 2. 1 つ以上のファイルをアーティファクトに追加する

アーティファクトメソッドを使用して、ファイル、ディレクトリ、外部 URI 参照 (Amazon S3 など) などを追加します。例えば、単一のテキストファイルを追加するには、[`add_file`]({{< relref path="/ref/python/artifact.md#add_file" lang="ja" >}}) メソッドを使用します：

```python
artifact.add_file(local_path="hello_world.txt", name="optional-name")
```

[`add_dir`]({{< relref path="/ref/python/artifact.md#add_dir" lang="ja" >}}) メソッドを使用して複数のファイルを追加することもできます。ファイルを追加する方法の詳細については、[アーティファクトを更新する]({{< relref path="./update-an-artifact.md" lang="ja" >}})を参照してください。

### 3. アーティファクトを W&B サーバーに保存する

最後に、アーティファクトを W&B サーバーに保存します。アーティファクトは run と関連付けられています。そのため、run オブジェクトの[`log_artifact()`]({{< relref path="/ref/python/run.md#log_artifact" lang="ja" >}})メソッドを使用してアーティファクトを保存します。

```python
# W&B Run を作成します。'job-type' を置き換えてください。
run = wandb.init(project="artifacts-example", job_type="job-type")

run.log_artifact(artifact)
```

W&B run の外部でアーティファクトを構築することもできます。詳細については、[外部ファイルを追跡する]({{< relref path="./track-external-files.md" lang="ja" >}})を参照してください。

{{% alert color="secondary" %}}
`log_artifact` への呼び出しは非同期に実行され、パフォーマンスの高いアップロードを実現します。これにより、ループでアーティファクトをログする際に驚くような振る舞いを引き起こす可能性があります。例えば：

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

アーティファクトのバージョン **v0** がそのメタデータ内でインデックス 0 を持つとは保証されません。アーティファクトは任意の順序でログされる可能性があるからです。
{{% /alert %}}

## アーティファクトにファイルを追加する

次のセクションでは、異なるファイルタイプと並列 run からアーティファクトを構築する方法を示します。

次の例では、複数のファイルとディレクトリ構造を持つプロジェクトディレクトリがあると仮定します：

```
project-directory
|-- images
|   |-- cat.png
|   +-- dog.png
|-- checkpoints
|   +-- model.h5
+-- model.h5
```

### 単一のファイルを追加

以下のコードスニペットは、単一のローカルファイルをアーティファクトに追加する方法を示しています：

```python
# 単一のファイルを追加
artifact.add_file(local_path="path/file.format")
```

例えば、作業中のローカルディレクトリに `'file.txt'` というファイルがあると仮定します：

```python
artifact.add_file("path/file.txt")  # `file.txt` として追加
```

アーティファクトの内容は次のようになります：

```
file.txt
```

オプションとして、アーティファクト内の所望のパスを `name` パラメータとして渡します。

```python
artifact.add_file(local_path="path/file.format", name="new/path/file.format")
```

アーティファクトは次のように保存されます：

```
new/path/file.txt
```

| API 呼び出し                                                | 結果のアーティファクト      |
| --------------------------------------------------------- | ------------------ |
| `artifact.add_file('model.h5')`                           | model.h5           |
| `artifact.add_file('checkpoints/model.h5')`               | model.h5           |
| `artifact.add_file('model.h5', name='models/mymodel.h5')` | models/mymodel.h5  |

### 複数のファイルを追加

次のコードスニペットは、ローカルディレクトリ全体をアーティファクトに追加する方法を示しています：

```python
# ディレクトリを再帰的に追加
artifact.add_dir(local_path="path/file.format", name="optional-prefix")
```

API 呼び出しにより次のアーティファクト内容が生成されます：

| API 呼び出し                                    | 結果のアーティファクト内容                                     |
| ------------------------------------------- | ------------------------------------------------------ |
| `artifact.add_dir('images')`                | <p><code>cat.png</code></p><p><code>dog.png</code></p> |
| `artifact.add_dir('images', name='images')` | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |
| `artifact.new_file('hello.txt')`            | `hello.txt`                                            |

### URI 参照を追加

アーティファクトは、URI が W&B ライブラリが処理方法を知っているスキームを持っている場合に、再現性のためのチェックサムやその他の情報を追跡します。

[`add_reference`]({{< relref path="/ref/python/artifact.md#add_reference" lang="ja" >}}) メソッドを使用して、アーティファクトに外部 URI 参照を追加します。`'uri'` 文字列を独自の URI に置き換えます。オプションとして、アーティファクト内の所望のパスを name パラメータとして渡します。

```python
# URI 参照を追加
artifact.add_reference(uri="uri", name="optional-name")
```

現在、アーティファクトは次の URI スキームをサポートしています：

* `http(s)://`: HTTP 経由でアクセス可能なファイルのパス。アーティファクトは、HTTP サーバーが `ETag` および `Content-Length` レスポンスヘッダーをサポートしている場合、エタグとサイズメタデータの形式でチェックサムを追跡します。
* `s3://`: S3 内のオブジェクトまたはオブジェクトプレフィックスへのパス。アーティファクトは、参照されたオブジェクトのチェックサムとバージョン管理情報 (バケットにオブジェクトバージョン管理が有効になっている場合) を追跡します。オブジェクトプレフィックスは、最大で 10,000 オブジェクトまで、プレフィックスの下のオブジェクトを含むように展開されます。
* `gs://`: GCS 内のオブジェクトまたはオブジェクトプレフィックスへのパス。アーティファクトは、参照されたオブジェクトのチェックサムとバージョン管理情報 (バケットにオブジェクトバージョン管理が有効になっている場合) を追跡します。オブジェクトプレフィックスは、最大で 10,000 オブジェクトまで、プレフィックスの下のオブジェクトを含むように展開されます。

以下の API 呼び出しにより、以下のアーティファクトが生成されます：

| API 呼び出し                                                                      | 結果のアーティファクト内容                                          |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `artifact.add_reference('s3://my-bucket/model.h5')`                           | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/checkpoints/model.h5')`               | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/model.h5', name='models/mymodel.h5')` | `models/mymodel.h5`                                                  |
| `artifact.add_reference('s3://my-bucket/images')`                             | <p><code>cat.png</code></p><p><code>dog.png</code></p>               |
| `artifact.add_reference('s3://my-bucket/images', name='images')`              | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |

### 並列 run からアーティファクトにファイルを追加する

大規模なデータセットや分散トレーニングでは、複数の並列 run が単一のアーティファクトに貢献する必要がある場合があります。

```python
import wandb
import time

# デモ目的のために並列で run を開始するために ray を使用します。
# 並列 run をオーケストレーションする方法は自由です。
import ray

ray.init()

artifact_type = "dataset"
artifact_name = "parallel-artifact"
table_name = "distributed_table"
parts_path = "parts"
num_parallel = 5

# 並列ライターの各バッチには、固有のグループ名が含まれている必要があります。
group_name = "writer-group-{}".format(round(time.time()))


@ray.remote
def train(i):
    """
    ライタージョブ。各ライターはアーティファクトに 1 つの画像を追加します。
    """
    with wandb.init(group=group_name) as run:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)

        # データを wandb テーブルに追加します。この場合、例のデータを使用します
        table = wandb.Table(columns=["a", "b", "c"], data=[[i, i * 2, 2**i]])

        # テーブルをアーティファクト内のフォルダに追加
        artifact.add(table, "{}/table_{}".format(parts_path, i))

        # アップサーティングは、アーティファクトを作成するかデータを追加します
        run.upsert_artifact(artifact)


# Run を並列で開始する
result_ids = [train.remote(i) for i in range(num_parallel)]

# すべてのライターのファイルが追加される前に、合流し完了を確認する
ray.get(result_ids)

# すべてのライターが終了したら、アーティファクトを完了して準備を示します。
with wandb.init(group=group_name) as run:
    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    # テーブルのフォルダを指す"PartitionTable" を作成し、アーティファクトに追加
    artifact.add(wandb.data_types.PartitionedTable(parts_path), table_name)

    # アーティファクトの完了は、このバージョンへの将来の "アップサート" を許可しません
    run.finish_artifact(artifact)
```