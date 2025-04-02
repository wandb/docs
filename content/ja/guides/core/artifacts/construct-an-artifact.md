---
title: Create an artifact
description: W&B の Artifact を作成、構築します。1 つまたは複数のファイル、または URI 参照を Artifact に追加する方法を学びます。
menu:
  default:
    identifier: ja-guides-core-artifacts-construct-an-artifact
    parent: artifacts
weight: 2
---

W&B Python SDK を使用して、[W&B Runs]({{< relref path="/ref/python/run.md" lang="ja" >}}) から Artifacts を構築します。[ファイル、ディレクトリー、URI、および並列 run からのファイルを Artifacts に追加]({{< relref path="#add-files-to-an-artifact" lang="ja" >}})できます。ファイルを Artifacts に追加したら、Artifacts を W&B サーバーまたは[独自のプライベートサーバー]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})に保存します。

Amazon S3 に保存されているファイルなど、外部ファイルを追跡する方法については、[外部ファイルを追跡]({{< relref path="./track-external-files.md" lang="ja" >}})のページを参照してください。

## Artifacts を構築する方法

[W&B Artifact]({{< relref path="/ref/python/artifact.md" lang="ja" >}})は、次の3つのステップで構築します。

### 1. `wandb.Artifact()` で Artifacts Python オブジェクトを作成する

[`wandb.Artifact()`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) クラスを初期化して、Artifacts オブジェクトを作成します。次の パラメータ を指定します。

*   **Name**: Artifacts の名前を指定します。名前は、一意で記述的で、覚えやすいものにする必要があります。Artifacts 名を使用して、W&B App UI で Artifacts を識別したり、その Artifacts を使用する場合に使用したりします。
*   **Type**: タイプを指定します。タイプは、シンプルで記述的で、機械学習 パイプライン の単一のステップに対応している必要があります。一般的な Artifacts タイプには、`'dataset'` または `'model'` があります。

{{% alert %}}
指定する「name」と「type」は、有向非巡回グラフの作成に使用されます。つまり、W&B App で Artifacts の リネージ を表示できます。

詳細については、[Artifacts グラフを探索およびトラバース]({{< relref path="./explore-and-traverse-an-artifact-graph.md" lang="ja" >}})を参照してください。
{{% /alert %}}

{{% alert color="secondary" %}}
タイプ パラメータ に別のタイプを指定した場合でも、Artifacts は同じ名前にすることはできません。つまり、タイプ `dataset` の `cats` という名前の Artifacts と、タイプ `model` の同じ名前の別の Artifacts を作成することはできません。
{{% /alert %}}

Artifacts オブジェクトを初期化するときに、オプションで説明と メタデータ を指定できます。利用可能な属性と パラメータ の詳細については、Python SDK リファレンス ガイドの [`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) クラス定義を参照してください。

次の例は、データセット Artifacts を作成する方法を示しています。

```python
import wandb

artifact = wandb.Artifact(name="<replace>", type="<replace>")
```

上記の コードスニペット の文字列 引数 を、自分の名前とタイプに置き換えます。

### 2. 1つ以上のファイルを Artifacts に追加する

Artifacts メソッド を使用して、ファイル、ディレクトリー、外部 URI 参照 (Amazon S3 など) などを追加します。たとえば、単一のテキスト ファイルを追加するには、[`add_file`]({{< relref path="/ref/python/artifact.md#add_file" lang="ja" >}}) メソッド を使用します。

```python
artifact.add_file(local_path="hello_world.txt", name="optional-name")
```

[`add_dir`]({{< relref path="/ref/python/artifact.md#add_dir" lang="ja" >}}) メソッド を使用して、複数のファイルを追加することもできます。ファイルを追加する方法の詳細については、[Artifacts を更新]({{< relref path="./update-an-artifact.md" lang="ja" >}})を参照してください。

### 3. Artifacts を W&B サーバー に保存する

最後に、Artifacts を W&B サーバー に保存します。Artifacts は run に関連付けられています。したがって、run オブジェクト の [`log_artifact()`]({{< relref path="/ref/python/run.md#log_artifact" lang="ja" >}}) メソッド を使用して、Artifacts を保存します。

```python
# W&B Run を作成します。'job-type' を置き換えます。
run = wandb.init(project="artifacts-example", job_type="job-type")

run.log_artifact(artifact)
```

オプションで、W&B run の外部で Artifacts を構築できます。詳細については、[外部ファイルを追跡]({{< relref path="./track-external-files.md" lang="ja" >}})を参照してください。

{{% alert color="secondary" %}}
`log_artifact` の呼び出しは、パフォーマンスの高いアップロードのために非同期的に実行されます。これにより、ループで Artifacts を ログ 記録するときに、予期しない 振る舞い が発生する可能性があります。次に例を示します。

```python
for i in range(10):
    a = wandb.Artifact(
        "race",
        type="dataset",
        metadata={
            "index": i,
        },
    )
    # ... Artifacts a にファイルを追加 ...
    run.log_artifact(a)
```

Artifacts バージョン **v0** は、Artifacts が任意の順序で ログ 記録される可能性があるため、 メタデータ に 0 のインデックスを持つことは保証されません。
{{% /alert %}}

## Artifacts にファイルを追加する

次のセクションでは、さまざまなファイル タイプ と並列 run から Artifacts を構築する方法について説明します。

次の例では、複数のファイルとディレクトリー構造を持つ プロジェクト ディレクトリー があると仮定します。

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

上記の コードスニペット は、単一のローカル ファイルを Artifacts に追加する方法を示しています。

```python
# 単一のファイルを追加
artifact.add_file(local_path="path/file.format")
```

たとえば、作業ローカル ディレクトリー に `'file.txt'` という名前のファイルがあるとします。

```python
artifact.add_file("path/file.txt")  # `file.txt' として追加されました
```

Artifacts には、次のコンテンツが含まれるようになりました。

```
file.txt
```

オプションで、`name` パラメータ に Artifacts 内の目的の パス を渡します。

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

上記の コードスニペット は、ローカル ディレクトリー 全体を Artifacts に追加する方法を示しています。

```python
# ディレクトリー を再帰的に追加
artifact.add_dir(local_path="path/file.format", name="optional-prefix")
```

上記の API 呼び出しは、上記の Artifacts コンテンツを生成します。

| API 呼び出し                                    | 結果の Artifacts                                     |
| ------------------------------------------- | ------------------------------------------------------ |
| `artifact.add_dir('images')`                | <p><code>cat.png</code></p><p><code>dog.png</code></p> |
| `artifact.add_dir('images', name='images')` | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |
| `artifact.new_file('hello.txt')`            | `hello.txt`                                            |

### URI 参照を追加する

Artifacts は、W&B ライブラリ が処理する方法を知っている スキーム を URI が持つ場合、再現性のために チェックサム やその他の情報を追跡します。

[`add_reference`]({{< relref path="/ref/python/artifact.md#add_reference" lang="ja" >}}) メソッド を使用して、外部 URI 参照を Artifacts に追加します。`'uri'` 文字列を自分の URI に置き換えます。オプションで、name パラメータ に Artifacts 内の目的の パス を渡します。

```python
# URI 参照を追加
artifact.add_reference(uri="uri", name="optional-name")
```

Artifacts は現在、次の URI スキーム をサポートしています。

*   `http(s)://`: HTTP 経由でアクセス可能なファイルへの パス 。Artifacts は、HTTP サーバー が `ETag` および `Content-Length` レスポンス ヘッダー をサポートしている場合、etag および サイズ メタデータ の形式で チェックサム を追跡します。
*   `s3://`: S3 内の オブジェクト または オブジェクト プレフィックス への パス 。Artifacts は、参照されている オブジェクト の チェックサム および バージョン 管理情報 ( バケット で オブジェクト の バージョン 管理が有効になっている場合) を追跡します。オブジェクト プレフィックス は、プレフィックス の下の オブジェクト を含むように拡張され、最大 10,000 個の オブジェクト が含まれます。
*   `gs://`: GCS 内の オブジェクト または オブジェクト プレフィックス への パス 。Artifacts は、参照されている オブジェクト の チェックサム および バージョン 管理情報 ( バケット で オブジェクト の バージョン 管理が有効になっている場合) を追跡します。オブジェクト プレフィックス は、プレフィックス の下の オブジェクト を含むように拡張され、最大 10,000 個の オブジェクト が含まれます。

上記の API 呼び出しは、上記の Artifacts を生成します。

| API 呼び出し                                                                      | 結果の Artifacts コンテンツ                                          |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `artifact.add_reference('s3://my-bucket/model.h5')`                           | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/checkpoints/model.h5')`               | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/model.h5', name='models/mymodel.h5')` | `models/mymodel.h5`                                                  |
| `artifact.add_reference('s3://my-bucket/images')`                             | <p><code>cat.png</code></p><p><code>dog.png</code></p>               |
| `artifact.add_reference('s3://my-bucket/images', name='images')`              | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |

### 並列 run から Artifacts にファイルを追加する

大規模な データセット または分散 トレーニング の場合、複数の並列 run が単一の Artifacts に貢献する必要がある場合があります。

```python
import wandb
import time

# デモ の目的で、ray を使用して並列で run を 起動 します。
# ただし、並列 run は、どのように編成してもかまいません。
import ray

ray.init()

artifact_type = "dataset"
artifact_name = "parallel-artifact"
table_name = "distributed_table"
parts_path = "parts"
num_parallel = 5

# 並列 ライター の各 バッチ には、独自の
# 一意の グループ 名が必要です。
group_name = "writer-group-{}".format(round(time.time()))


@ray.remote
def train(i):
    """
    ライター ジョブ 。各 ライター は、1つの画像を Artifacts に追加します。
    """
    with wandb.init(group=group_name) as run:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)

        # データを wandb テーブル に追加します。この場合、サンプル データ を使用します
        table = wandb.Table(columns=["a", "b", "c"], data=[[i, i * 2, 2**i]])

        # テーブル を Artifacts 内の フォルダ に追加します
        artifact.add(table, "{}/table_{}".format(parts_path, i))

        # Artifacts を アップサート すると、Artifacts のデータ が作成または追加されます
        run.upsert_artifact(artifact)


# 並列で run を 起動 します
result_ids = [train.remote(i) for i in range(num_parallel)]

# すべての ライター に 参加 して、ファイル が
# Artifacts を終了する前に追加されていることを確認します。
ray.get(result_ids)

# すべての ライター が終了したら、Artifacts を終了して、
# 準備完了にします。
with wandb.init(group=group_name) as run:
    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    # テーブル の フォルダ を指す "PartitionTable" を作成します
    # Artifacts に追加します。
    artifact.add(wandb.data_types.PartitionedTable(parts_path), table_name)

    # Finish artifact は Artifacts を確定し、この バージョン への将来の "upsert" を許可しません
    run.finish_artifact(artifact)
```