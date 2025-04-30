---
title: アーティファクトを作成する
description: W&B アーティファクトを作成し、構築します。ファイルや URI リファレンスをアーティファクトに追加する方法を学びましょう。
menu:
  default:
    identifier: ja-guides-core-artifacts-construct-an-artifact
    parent: artifacts
weight: 2
---

W&B Python SDK を使用して、[W&B Runs]({{< relref path="/ref/python/run.md" lang="ja" >}}) から artifacts を構築します。[ファイル、ディレクトリ、URI、並行実行からのファイルを artifacts に追加]({{< relref path="#add-files-to-an-artifact" lang="ja" >}})できます。ファイルをアーティファクトに追加した後、W&B サーバーまたは[自分のプライベートサーバー]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})に保存します。

Amazon S3 に保存されているファイルなど、外部ファイルのトラッキング方法については、[外部ファイルのトラッキング]({{< relref path="./track-external-files.md" lang="ja" >}})ページをご覧ください。

## アーティファクトの構築方法

3 つのステップで [W&B Artifact]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) を構築します。

### 1. `wandb.Artifact()` でアーティファクト Python オブジェクトを作成する

[`wandb.Artifact()`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) クラスを初期化して、アーティファクトオブジェクトを作成します。以下のパラメータを指定します。

* **Name**: アーティファクトの名前を指定します。名前は一意で説明的、かつ記憶しやすいものにしてください。Artifacts の名前は、W&B アプリ UI でアーティファクトを識別するとき、またそのアーティファクトを使用したいときに使用します。
* **Type**: タイプを指定します。タイプはシンプルで説明的で、機械学習パイプラインの単一ステップに対応するものでなければなりません。一般的なアーティファクトタイプには `'dataset'` や `'model'` があります。

{{% alert %}}
指定した「name」と「type」は、有向非循環グラフを作成するために使用されます。これは、W&B アプリでアーティファクトのリネージを見ることができることを意味します。

詳細については、[アーティファクトグラフの探索とトラバース]({{< relref path="./explore-and-traverse-an-artifact-graph.md" lang="ja" >}})をご覧ください。
{{% /alert %}}


{{% alert color="secondary" %}}
Artifacts は、たとえ異なる types パラメータを指定しても、同じ名前を持つことはできません。つまり、`cats` という名前のタイプ `dataset` のアーティファクトを作成し、同じ名前のタイプ `model` のアーティファクトを作成することはできません。
{{% /alert %}}

アーティファクトオブジェクトを初期化する際に、オプションで説明とメタデータを提供できます。利用可能な属性とパラメータの詳細については、Python SDK Reference Guide の [`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) クラス定義をご覧ください。

次の例は、データセットアーティファクトを作成する方法を示しています。

```python
import wandb

artifact = wandb.Artifact(name="<replace>", type="<replace>")
```

先行のコードスニペット内の文字列の引数を、自分の 固有の名前とタイプで置き換えてください。

### 2. アーティファクトに 1 つ以上のファイルを追加する

ファイル、ディレクトリ、外部 URI リファレンス（例: Amazon S3）などをアーティファクトメソッドで追加します。たとえば、単一のテキストファイルを追加するには、[`add_file`]({{< relref path="/ref/python/artifact.md#add_file" lang="ja" >}}) メソッドを使用します。

```python
artifact.add_file(local_path="hello_world.txt", name="optional-name")
```

また、複数のファイルを [`add_dir`]({{< relref path="/ref/python/artifact.md#add_dir" lang="ja" >}}) メソッドで追加することもできます。ファイルを追加する方法の詳細については、[アーティファクトの更新]({{< relref path="./update-an-artifact.md" lang="ja" >}})をご覧ください。

### 3. アーティファクトを W&B サーバーに保存する

最後に、アーティファクトを W&B サーバーに保存します。Artifacts は run に関連付けられます。したがって、run オブジェクトの [`log_artifact()`]({{< relref path="/ref/python/run.md#log_artifact" lang="ja" >}}) メソッドを使用してアーティファクトを保存します。

```python
# W&B Run を作成します。'job-type' を置き換えます。
run = wandb.init(project="artifacts-example", job_type="job-type")

run.log_artifact(artifact)
```

W&B Run の外でアーティファクトを作成することもできます。詳細については、[外部ファイルのトラッキング]({{< relref path="./track-external-files.md" lang="ja" >}})をご覧ください。

{{% alert color="secondary" %}}
`log_artifact` の呼び出しは、パフォーマンスを向上させるために非同期で行われます。これにより、ループ内でアーティファクトをログする際に予期しない挙動が生じることがあります。たとえば、

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

アーティファクトバージョン **v0** は、そのメタデータにインデックス 0 があることを保証しません。アーティファクトは任意の順序でログされる可能性があるためです。
{{% /alert %}}

## アーティファクトにファイルを追加

以下のセクションでは、異なるファイルタイプおよび並行実行からのアーティファクトを構築する方法を説明します。

以下の例では、複数のファイルとディレクトリ構造を持つプロジェクトディレクトリを前提とします。

```
project-directory
|-- images
|   |-- cat.png
|   +-- dog.png
|-- checkpoints
|   +-- model.h5
+-- model.h5
```

### 単一ファイルを追加

以下のコードスニペットは、ローカルの単一ファイルをアーティファクトに追加する方法を示しています。

```python
# 単一のファイルを追加
artifact.add_file(local_path="path/file.format")
```

たとえば、作業中のローカルディレクトリに `'file.txt'` というファイルがあるとします。

```python
artifact.add_file("path/file.txt")  # `file.txt` として追加されました
```

アーティファクトは次の内容を持つようになります。

```
file.txt
```

オプションで、アーティファクト内の `name` パラメータの希望するパスを渡して下さい。

```python
artifact.add_file(local_path="path/file.format", name="new/path/file.format")
```

アーティファクトは次のように保存されます。

```
new/path/file.txt
```

| API Call                                                  | Resulting artifact |
| --------------------------------------------------------- | ------------------ |
| `artifact.add_file('model.h5')`                           | model.h5           |
| `artifact.add_file('checkpoints/model.h5')`               | model.h5           |
| `artifact.add_file('model.h5', name='models/mymodel.h5')` | models/mymodel.h5  |

### 複数ファイルを追加

以下のコードスニペットは、ローカルのディレクトリ全体をアーティファクトに追加する方法を示しています。

```python
# ディレクトリを再帰的に追加
artifact.add_dir(local_path="path/file.format", name="optional-prefix")
```

以下の API 呼び出しは、以下のアーティファクトコンテンツを生成します。

| API Call                                    | Resulting artifact                                     |
| ------------------------------------------- | ------------------------------------------------------ |
| `artifact.add_dir('images')`                | <p><code>cat.png</code></p><p><code>dog.png</code></p> |
| `artifact.add_dir('images', name='images')` | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |
| `artifact.new_file('hello.txt')`            | `hello.txt`                                            |

### URI リファレンスを追加

アーティファクトは、URI が W&B ライブラリが扱えるスキームを持つ場合、再現性のためにチェックサムやその他の情報をトラッキングします。

[`add_reference`]({{< relref path="/ref/python/artifact.md#add_reference" lang="ja" >}}) メソッドを使用して、アーティファクトに外部 URI リファレンスを追加します。 `'uri'` 文字列を自分の URI で置き換えてください。オプションで、アーティファクト内の `name` パラメータの希望するパスを渡して下さい。

```python
# URI リファレンスを追加
artifact.add_reference(uri="uri", name="optional-name")
```

現在、アーティファクトは以下の URI スキームをサポートしています。

* `http(s)://`: HTTP 上でアクセス可能なファイルへのパス。HTTP サーバーが `ETag` や `Content-Length` レスポンスヘッダーをサポートしている場合、アーティファクトはエタグとサイズメタデータの形でチェックサムをトラッキングします。
* `s3://`: S3 内のオブジェクトまたはオブジェクトプレフィックスへのパス。アーティファクトは、参照されたオブジェクトのためのチェックサムとバージョン管理情報（バケットにオブジェクトバージョン管理が有効になっている場合）をトラッキングします。オブジェクトプレフィックスは、プレフィックスの下にあるオブジェクトを最大 10,000 個まで含むように展開されます。
* `gs://`: GCS 内のオブジェクトまたはオブジェクトプレフィックスへのパス。アーティファクトは、参照されたオブジェクトのためのチェックサムとバージョン管理情報（バケットにオブジェクトバージョン管理が有効になっている場合）をトラッキングします。オブジェクトプレフィックスは、プレフィックスの下にあるオブジェクトを最大 10,000 個まで含むように展開されます。

以下の API 呼び出しは、以下のアーティファクトを生成します。

| API call                                                                      | Resulting artifact contents                                          |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `artifact.add_reference('s3://my-bucket/model.h5')`                           | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/checkpoints/model.h5')`               | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/model.h5', name='models/mymodel.h5')` | `models/mymodel.h5`                                                  |
| `artifact.add_reference('s3://my-bucket/images')`                             | <p><code>cat.png</code></p><p><code>dog.png</code></p>               |
| `artifact.add_reference('s3://my-bucket/images', name='images')`              | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |

### 並行実行からアーティファクトにファイルを追加

大規模な datasets または分散トレーニングの場合、複数の並行 run が単一のアーティファクトに貢献する必要があるかもしれません。

```python
import wandb
import time

# デモンストレーションのために、run を並行して起動するために ray を使用します。
# 並行 run は任意の方法で調整できます。
import ray

ray.init()

artifact_type = "dataset"
artifact_name = "parallel-artifact"
table_name = "distributed_table"
parts_path = "parts"
num_parallel = 5

# 並行ライターの各バッチは、その独自の
# グループ名を持つ必要があります。
group_name = "writer-group-{}".format(round(time.time()))


@ray.remote
def train(i):
    """
    各ライターは、アーティファクトに 1 つの画像を追加します。
    """
    with wandb.init(group=group_name) as run:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)

        # wandb テーブルにデータを追加します。この場合、例としてデータを使用します。
        table = wandb.Table(columns=["a", "b", "c"], data=[[i, i * 2, 2**i]])

        # アーティファクト内のフォルダーにテーブルを追加します。
        artifact.add(table, "{}/table_{}".format(parts_path, i))

        # アーティファクトをアップサーティングすることで、アーティファクトにデータを作成または追加します。
        run.upsert_artifact(artifact)


# 並行して run を起動
result_ids = [train.remote(i) for i in range(num_parallel)]

# ファイルがアーティファクトに追加されたことを確認するために
# すべてのライターを待機します。
ray.get(result_ids)

# すべてのライターが終了したら、アーティファクトを終了して
# 使用可能であることをマークします。
with wandb.init(group=group_name) as run:
    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    # "PartitionTable" を作成し、フォルダーのテーブルを指すようにして
    # アーティファクトに追加します。
    artifact.add(wandb.data_types.PartitionedTable(parts_path), table_name)

    # アーティファクトを終了することで、このバージョンへの
    # 将来の"upserts"を禁止します。
    run.finish_artifact(artifact)
```