---
title: アーティファクトを作成する
description: W&B Artifact を作成・構築します。Artifact にファイルや URI 参照を追加する方法を学びましょう。
menu:
  default:
    identifier: construct-an-artifact
    parent: artifacts
weight: 2
---

W&B Python SDK を使って [W&B Runs]({{< relref "/ref/python/sdk/classes/run.md" >}}) からアーティファクトを作成できます。[ファイル、ディレクトリ、URI、または並列 Run からのファイルをアーティファクトに追加]({{< relref "#add-files-to-an-artifact" >}}) することが可能です。ファイルをアーティファクトに追加したら、そのアーティファクトを W&B サーバ、もしくは [自分のプライベートサーバ]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) へ保存しましょう。

Amazon S3 などに保存されている外部ファイルのトラッキングについては、[外部ファイルのトラッキング]({{< relref "./track-external-files.md" >}}) のページをご覧ください。

## アーティファクトの作成方法

[W&B Artifact]({{< relref "/ref/python/sdk/classes/artifact.md" >}}) は以下の３ステップで作成します。

### 1. `wandb.Artifact()` でアーティファクトの Python オブジェクトを作成

[`wandb.Artifact()`]({{< relref "/ref/python/sdk/classes/artifact.md" >}}) クラスを初期化してアーティファクトオブジェクトを作成します。以下のパラメータを指定してください。

* **Name**: アーティファクトの名前を指定します。この名前は一意で分かりやすく、記憶しやすいものにしてください。この名前は W&B App UI でアーティファクトを識別したり、アーティファクトを使用する際にも利用します。
* **Type**: タイプを指定します。タイプはシンプルで分かりやすく、機械学習パイプラインの単一ステップに対応するものがよいです。一般的なタイプには `'dataset'` や `'model'` があります。

{{% alert %}}
指定した "name" と "type" は、有向非巡回グラフ（DAG）作成に利用されます。つまり、W&B App でアーティファクトのリネージ（由来・系譜）を確認できます。

詳細は [アーティファクトグラフの探索とトラバース方法]({{< relref "./explore-and-traverse-an-artifact-graph.md" >}}) をご覧ください。
{{% /alert %}}

{{% alert color="secondary" %}}
アーティファクトは、たとえ type パラメータが異なっていても同じ名前を共有できません。つまり、`dataset` タイプの `cats` アーティファクトと、`model` タイプの `cats` アーティファクトの両方は作成できません。
{{% /alert %}}

アーティファクトオブジェクトの初期化時に、説明やメタデータもオプションで指定できます。利用可能な属性やパラメータの詳細は、Python SDK リファレンスガイドの [`wandb.Artifact`]({{< relref "/ref/python/sdk/classes/artifact.md" >}}) クラス定義を参照してください。

以下は dataset アーティファクトの作成例です。

```python
import wandb

artifact = wandb.Artifact(name="<replace>", type="<replace>")
```

上記コードスニペットの引数を自分の名前とタイプに置き換えてください。

### 2. アーティファクトに 1 つ以上のファイルを追加

アーティファクトのメソッドを使って、ファイル、ディレクトリ、外部 URI（例: Amazon S3 など）を追加できます。たとえば、テキストファイルを 1 つ追加するには [`add_file`]({{< relref "/ref/python/sdk/classes/artifact.md#add_file" >}}) メソッドを利用します。

```python
artifact.add_file(local_path="hello_world.txt", name="optional-name")
```

複数ファイルを追加したい場合、[`add_dir`]({{< relref "/ref/python/sdk/classes/artifact.md#add_dir" >}}) メソッドでディレクトリごと追加できます。ファイル追加の詳細は [アーティファクトの更新]({{< relref "./update-an-artifact.md" >}}) を参照してください。

### 3. アーティファクトを W&B サーバに保存

最後に、アーティファクトを W&B サーバへ保存します。アーティファクトは Run と関連付けられるため、Run オブジェクトの [`log_artifact()`]({{< relref "/ref/python/sdk/classes/run.md#log_artifact" >}}) メソッドを使って保存します。

```python
# W&B Run を作成。'job-type' を自身の用途に合わせて変更してください
run = wandb.init(project="artifacts-example", job_type="job-type")

run.log_artifact(artifact)
```

また、W&B Run の外でアーティファクトを作成することも可能です。詳しくは [外部ファイルのトラッキング]({{< relref "./track-external-files.md" >}}) をご覧ください。

{{% alert color="secondary" %}}
`log_artifact` の呼び出しはパフォーマンス向上のため非同期で実行されます。これにより、ループ内でアーティファクトをログすると想定外の挙動になる場合があります。例として：

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

アーティファクトバージョン **v0** の metadata の index が 0 である保証はなく、アーティファクトは任意の順序でログされる可能性があります。
{{% /alert %}}

## アーティファクトへファイルを追加

ここからは、異なるファイルタイプや並列 Run からアーティファクトを構築する方法を説明します。

以下の例では、複数ファイルがあるプロジェクトディレクトリとディレクトリ構造を前提とします。

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

以下のコードスニペットでは、ローカルの単一ファイルをアーティファクトへ追加する方法を示しています。

```python
# ファイルをひとつ追加
artifact.add_file(local_path="path/file.format")
```

例えば、作業ディレクトリ内に `'file.txt'` というファイルがある場合：

```python
artifact.add_file("path/file.txt")  # `file.txt` として追加されます
```

このアーティファクトの内容は次のようになります。

```
file.txt
```

name パラメータでアーティファクト内の格納先パスも指定できます。

```python
artifact.add_file(local_path="path/file.format", name="new/path/file.format")
```

この場合はアーティファクト内では次のように保存されます。

```
new/path/file.txt
```

| API 呼び出し                                        | 結果となるアーティファクト             |
| --------------------------------------------------- | --------------------- |
| `artifact.add_file('model.h5')`                     | model.h5              |
| `artifact.add_file('checkpoints/model.h5')`         | model.h5              |
| `artifact.add_file('model.h5', name='models/mymodel.h5')` | models/mymodel.h5     |

### 複数ファイルの追加

以下は、ローカルディレクトリ全体を再帰的にアーティファクトへ追加する方法です。

```python
# ディレクトリを再帰的に追加
artifact.add_dir(local_path="path/file.format", name="optional-prefix")
```

これらの API 呼び出しは以下のようなアーティファクトを生成します。

| API 呼び出し                             | 結果となるアーティファクト内容                              |
| ---------------------------------------- | -------------------------------------------------- |
| `artifact.add_dir('images')`             | <p><code>cat.png</code></p><p><code>dog.png</code></p> |
| `artifact.add_dir('images', name='images')` | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |
| `artifact.new_file('hello.txt')`         | `hello.txt`                                        |

### URI 参照の追加

Artifacts は、その URI のスキームを W&B ライブラリが認識できる場合、再現性確保のためにチェックサムや諸情報をトラッキングします。

外部 URI 参照をアーティファクトへ追加する際は、[`add_reference`]({{< relref "/ref/python/sdk/classes/artifact.md#add_reference" >}}) メソッドを利用してください。'uri' 部分を自身の URI で置き換え、name パラメータでアーティファクト内の格納先パスを指定できます（任意）。

```python
# URI 参照を追加
artifact.add_reference(uri="uri", name="optional-name")
```

Artifacts では、以下の URI スキームが現在サポートされています。

* `http(s)://`: HTTP 経由でアクセス可能なファイルパス。HTTP サーバが `ETag` や `Content-Length` レスポンスヘッダをサポートしていれば、artifact は etag やサイズのメタデータをトラッキングします。
* `s3://`: S3 のオブジェクトまたはプレフィックスパス。バケットがオブジェクトバージョニングを有効にしていれば、artifact では対象オブジェクトのチェックサムとバージョン情報を追跡します。プレフィックスは最大 10,000 オブジェクトまで展開可能です。
* `gs://`: GCS のオブジェクトまたはプレフィックスパス。バケットがオブジェクトバージョニングを有効にしていれば、artifact では対象オブジェクトのチェックサムとバージョン情報を追跡します。プレフィックスは最大 10,000 オブジェクトまで展開可能です。

以下の API 呼び出しによるアーティファクト例:

| API 呼び出し                                                        | 結果となるアーティファクト内容                                       |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `artifact.add_reference('s3://my-bucket/model.h5')`                    | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/checkpoints/model.h5')`        | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/model.h5', name='models/mymodel.h5')` | `models/mymodel.h5`                                                  |
| `artifact.add_reference('s3://my-bucket/images')`                      | <p><code>cat.png</code></p><p><code>dog.png</code></p>               |
| `artifact.add_reference('s3://my-bucket/images', name='images')`       | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |

### 並列 Run からアーティファクトへファイルを追加

大規模データセットや分散トレーニングの際は、複数の並列 Run から 1 つのアーティファクトへファイルを追加することができます。

```python
import wandb
import time

# この例では ray で Run を並列起動します。
# 好きな方法で並列 Run をオーケストレーションしてかまいません。
import ray

ray.init()

artifact_type = "dataset"
artifact_name = "parallel-artifact"
table_name = "distributed_table"
parts_path = "parts"
num_parallel = 5

# 並列書き込みバッチごとに一意な group 名を持たせます。
group_name = "writer-group-{}".format(round(time.time()))


@ray.remote
def train(i):
    """
    writer のジョブ。各 writer がひとつずつ画像をアーティファクトへ追加。
    """
    with wandb.init(group=group_name) as run:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)

        # サンプルデータを利用して wandb table にデータを追加
        table = wandb.Table(columns=["a", "b", "c"], data=[[i, i * 2, 2**i]])

        # artifact 内のフォルダに table を追加
        artifact.add(table, "{}/table_{}".format(parts_path, i))

        # upsert で artifact の新規作成またはデータを追加
        run.upsert_artifact(artifact)


# 並列で Run を起動
result_ids = [train.remote(i) for i in range(num_parallel)]

# 全 writer の処理完了を待機
ray.get(result_ids)

# 全 writer 終了後、artifact を finalize
# これで artifact が変更不可となります。
with wandb.init(group=group_name) as run:
    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    # tables フォルダを指す "PartitionTable" を作成し artifact に追加
    artifact.add(wandb.data_types.PartitionedTable(parts_path), table_name)

    # finish_artifact で artifact のバージョンを確定。以後 upsert 不可。
    run.finish_artifact(artifact)
```