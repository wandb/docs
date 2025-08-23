---
title: アーティファクトを作成
description: W&B Artifact を作成・構築する方法をご紹介します。Artifact に 1 つ以上のファイルや URI 参照を追加する方法について学びましょう。
menu:
  default:
    identifier: ja-guides-core-artifacts-construct-an-artifact
    parent: artifacts
weight: 2
---

W&B Python SDK を使って [W&B Runs]({{< relref path="/ref/python/sdk/classes/run.md" lang="ja" >}}) から Artifacts を作成できます。[ファイル、ディレクトリー、URI、そして並列実行からのファイルも Artifacts に追加できます]({{< relref path="#add-files-to-an-artifact" lang="ja" >}})。ファイルを Artifact に追加した後は、その Artifact を W&B サーバーや[独自のプライベートサーバー]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) に保存しましょう。

Amazon S3 など外部ファイルのトラッキング方法については、[外部ファイルをトラッキングする]({{< relref path="./track-external-files.md" lang="ja" >}}) ページをご参照ください。

## Artifact の作成方法

[W&B Artifact]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) は以下の 3 ステップで作成します。

### 1. `wandb.Artifact()` で Artifact の Python オブジェクトを作成

[`wandb.Artifact()`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) クラスを初期化して Artifact オブジェクトを作成します。下記パラメータを指定してください：

* **Name**: Artifact の名前を指定します。名前は一意で、分かりやすく、覚えやすいものにしましょう。Artifact の名前は、W&B App UI でも識別に使いますし、その Artifact を利用したいときにも必要です。
* **Type**: タイプを指定します。タイプはシンプルかつ分かりやすく、機械学習パイプラインの 1 ステップに対応するものにしてください。よく使われるタイプ例は `'dataset'` や `'model'` です。

{{% alert %}}
指定した "name" と "type" から有向非巡回グラフ（DAG）が作成されます。これにより、W&B App 上で Artifact のリネージを可視化できます。

詳細は [Artifact グラフの探索とトラバース]({{< relref path="./explore-and-traverse-an-artifact-graph.md" lang="ja" >}}) をご覧ください。
{{% /alert %}}

{{% alert color="secondary" %}}
Artifacts は同じ名前を持つことはできません。たとえ types パラメータが異なる場合でも同名は不可です。例えば、`cats` という名前の `dataset` タイプの Artifact と、同じく `cats` という名前の `model` タイプの Artifact を両方作成することはできません。
{{% /alert %}}

Artifact オブジェクトを初期化する際に、説明やメタデータもオプションで追加できます。利用可能な属性やパラメータについては、Python SDK リファレンスガイドの [`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) クラス定義をご参照ください。

以下はデータセット Artifact を作成するサンプル例です：

```python
import wandb

artifact = wandb.Artifact(name="<replace>", type="<replace>")
```

上記コードスニペットの文字列部分を、ご自身の名前とタイプに置き換えてください。

### 2. Artifact にファイルを追加

Artifact のメソッドを使って、ファイルやディレクトリー、外部 URI 参照（例えば Amazon S3 など）を追加できます。例えば、テキストファイルを 1 つ追加する場合は、[`add_file`]({{< relref path="/ref/python/sdk/classes/artifact.md#add_file" lang="ja" >}}) メソッドを利用します:

```python
artifact.add_file(local_path="hello_world.txt", name="optional-name")
```

複数ファイルを追加したい場合は、[`add_dir`]({{< relref path="/ref/python/sdk/classes/artifact.md#add_dir" lang="ja" >}}) メソッドが利用できます。ファイル追加についての詳細は [Artifact の更新]({{< relref path="./update-an-artifact.md" lang="ja" >}}) をご覧ください。

### 3. Artifact を W&B サーバーに保存

最後に、Artifact を W&B サーバーに保存します。Artifacts は Run に紐づきますので、Run オブジェクトの [`log_artifact()`]({{< relref path="/ref/python/sdk/classes/run.md#log_artifact" lang="ja" >}}) メソッドを使って Artifact を保存します。

```python
# W&B Run を作成します。'job-type' はご自身で置き換えてください
run = wandb.init(project="artifacts-example", job_type="job-type")

run.log_artifact(artifact)
```

W&B Run の外で Artifact を作成することも可能です。詳細は [外部ファイルをトラッキングする]({{< relref path="./track-external-files.md" lang="ja" >}}) をご覧ください。

{{% alert color="secondary" %}}
`log_artifact` の呼び出しは、パフォーマンスのため非同期でアップロードされます。これにより、Artifact をループでログする場合、予期しない順序で登録されることがあります。例：

```python
for i in range(10):
    a = wandb.Artifact(
        "race",
        type="dataset",
        metadata={
            "index": i,
        },
    )
    # ... Artifact a にファイルを追加 ...
    run.log_artifact(a)
```

Artifact バージョン **v0** の metadata の index が必ずしも 0 になるとは限りません。Artifact は任意の順序でログされる可能性があります。
{{% /alert %}}

## Artifact にファイルを追加

以下のセクションでは、さまざまなファイルタイプや並列実行から Artifact を作成する方法について説明します。

例として、複数ファイルとディレクトリー構造を持つプロジェクトディレクトリがあるとします：

```
project-directory
|-- images
|   |-- cat.png
|   +-- dog.png
|-- checkpoints
|   +-- model.h5
+-- model.h5
```

### 単一ファイルの追加

下記のコードスニペットは、ローカルの単一ファイルを Artifact に追加する方法を示します：

```python
# ファイル1つを追加
artifact.add_file(local_path="path/file.format")
```

たとえば、`'file.txt'` というファイルがローカル作業ディレクトリーにある場合：

```python
artifact.add_file("path/file.txt")  # `file.txt` として追加
```

このとき Artifact の内容は以下のようになります：

```
file.txt
```

`name` パラメータに、Artifact 内での希望パスを指定することもできます。

```python
artifact.add_file(local_path="path/file.format", name="new/path/file.format")
```

この場合 Artifact には次のように格納されます：

```
new/path/file.txt
```

| API Call                                                  | Artifact の内容         |
| --------------------------------------------------------- | ----------------------- |
| `artifact.add_file('model.h5')`                           | model.h5                |
| `artifact.add_file('checkpoints/model.h5')`               | model.h5                |
| `artifact.add_file('model.h5', name='models/mymodel.h5')` | models/mymodel.h5       |

### 複数ファイルの追加

下記のコードスニペットは、ローカルのディレクトリー全体を再帰的に Artifact に追加するサンプルです：

```python
# ディレクトリーを再帰的に追加
artifact.add_dir(local_path="path/file.format", name="optional-prefix")
```

この API 呼び出しを行うことで、以下のような内容の Artifact が作成されます：

| API Call                                    | Artifact の内容                                            |
| ------------------------------------------- | ---------------------------------------------------------- |
| `artifact.add_dir('images')`                | <p><code>cat.png</code></p><p><code>dog.png</code></p>     |
| `artifact.add_dir('images', name='images')` | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |
| `artifact.new_file('hello.txt')`            | `hello.txt`                                                |

### URI リファレンスの追加

URI の scheme が W&B ライブラリで認識される場合、Artifacts は再現性のためチェックサムなどの情報を追跡します。

外部 URI リファレンスは [`add_reference`]({{< relref path="/ref/python/sdk/classes/artifact.md#add_reference" lang="ja" >}}) メソッドで追加します。`'uri'` をご自身の URI に置き換えてください。必要なら name パラメータで Artifact 内の希望パスも指定可能です。

```python
# URI リファレンスを追加
artifact.add_reference(uri="uri", name="optional-name")
```

Artifacts は現在以下の URI scheme をサポートしています：

* `http(s)://`：HTTP 経由でアクセス可能なファイルのパス。 Artifact は、HTTP サーバーが `ETag` および `Content-Length` レスポンスヘッダーに対応していれば、etag やサイズのメタデータとしてチェックサムを記録します。
* `s3://`：S3 上のオブジェクトやプレフィックスへのパス。バケットでオブジェクトバージョン管理が有効なら、チェックサムおよびバージョン情報も追跡します。オブジェクトプレフィックスは下層最大 10,000個オブジェクトまでが展開されます。
* `gs://`：GCS 上のオブジェクトまたはそのプレフィックスへのパス。S3 同様、最大 10,000個まで対応しています。

以下の API 呼び出し例では、各 Artifact に以下の内容が追加されます：

| API call                                                                      | Artifact に入る内容                                                |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `artifact.add_reference('s3://my-bucket/model.h5')`                           | `model.h5`                                                         |
| `artifact.add_reference('s3://my-bucket/checkpoints/model.h5')`               | `model.h5`                                                         |
| `artifact.add_reference('s3://my-bucket/model.h5', name='models/mymodel.h5')` | `models/mymodel.h5`                                                |
| `artifact.add_reference('s3://my-bucket/images')`                             | <p><code>cat.png</code></p><p><code>dog.png</code></p>             |
| `artifact.add_reference('s3://my-bucket/images', name='images')`              | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |

### 並列 Run から Artifact へファイルを追加

大規模なデータセットや分散トレーニングの場合、複数の並列 Run で 1 つの Artifact にデータを加えることがあります。

```python
import wandb
import time

# この例では ray を使って並列で Run を立ち上げます
# （お好きな並列化方法を使ってください）
import ray

ray.init()

artifact_type = "dataset"
artifact_name = "parallel-artifact"
table_name = "distributed_table"
parts_path = "parts"
num_parallel = 5

# 並列 writer のバッチごとに一意な group 名が必要です
group_name = "writer-group-{}".format(round(time.time()))

@ray.remote
def train(i):
    """
    writer job です。各 writer が Artifact に 1 つずつ画像を追加します。
    """
    with wandb.init(group=group_name) as run:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)

        # 例のデータで wandb table を追加
        table = wandb.Table(columns=["a", "b", "c"], data=[[i, i * 2, 2**i]])

        # Artifact のフォルダーにテーブルを追加
        artifact.add(table, "{}/table_{}".format(parts_path, i))

        # Artifact の upsert で作成・データを append
        run.upsert_artifact(artifact)

# 並列で Run を立ち上げ
result_ids = [train.remote(i) for i in range(num_parallel)]

# すべての writer のファイルが追加されるまで待機
ray.get(result_ids)

# すべて writer が終わったら、Artifact を finalize します
with wandb.init(group=group_name) as run:
    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    # テーブルのフォルダーを指す "PartitionTable" を作成し、
    # Artifact に追加
    artifact.add(wandb.data_types.PartitionedTable(parts_path), table_name)

    # finish_artifact で Artifact を finalize。今後の upsert は不可
    run.finish_artifact(artifact)
```