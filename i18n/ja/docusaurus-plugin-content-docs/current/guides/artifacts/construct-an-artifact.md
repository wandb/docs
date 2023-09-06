---
description: >-
  Create, construct a W&B Artifact. Learn how to add one or more files or a URI
  reference to an Artifact.
displayed_sidebar: ja
---

# アーティファクトの構築

<head>
  <title>アーティファクトの構築</title>
</head>
Weights & Biases SDKを使って、[Weights & Biases Run](https://docs.wandb.ai/ref/python/run)中または外でアーティファクトを構築します。ファイル、ディレクトリ、URI、並列実行からのファイルをアーティファクトに追加します。ファイルやURIが追加されたら、W&B Runを使ってWeights & Biasesにアーティファクトを保存します。Weights & Biases Runの外部で外部ファイルをトラッキングする方法については、[外部ファイルのトラッキング](https://docs.wandb.ai/guides/artifacts/track-external-files)を参照してください。

### アーティファクトの構築方法

Run内での[Weights & Biases Artifact](https://docs.wandb.ai/ref/python/artifact)の構築は、3つのステップで行われます:
#### 1. `wandb.Artifact()`を使って、アーティファクトのPythonオブジェクトを作成する

[`wandb.Artifact()`](https://docs.wandb.ai/ref/python/artifact) クラスを初期化して、アーティファクトを作成します。以下のパラメータを指定してください。

* **名前**: アーティファクトに一意でわかりやすく記述的な名前を指定してください。アーティファクトの名前は、W&BアプリUIでアーティファクトを識別するときと、そのアーティファクトを使用したいときに使います。
* **タイプ**: タイプを指定してください。タイプはシンプルで記述的で、機械学習の開発フローの単一のステップに対応するべきです。一般的なアーティファクトのタイプには、`'dataset'` や `'model'` があります。

W&BアプリUI内でアーティファクトの履歴を表示することができます。提供した "名前" と "タイプ" は、有向非巡回グラフを作成するために使用されます。詳細については、[アーティファクトグラフの探索と移動](./explore-and-traverse-an-artifact-graph.md)を参照してください。
:::caution
アーティファクトは、同じ名前を持つことができません。たとえ、typesパラメーターに異なるタイプを指定したとしてもです。つまり、「dataset」タイプの「cats」という名前のアーティファクトと、「model」タイプの同じ名前のアーティファクトを作成することはできません。
:::

アーティファクトオブジェクトを初期化する際に、オプションで説明とメタデータを提供できます。利用可能な属性やパラメータの詳細については、Python SDKリファレンスガイドの[wandb.Artifact](https://docs.wandb.ai/ref/python/artifact)クラス定義を参照してください。

次の例では、データセットアーティファクトの作成方法について説明しています：

```python
import wandb

artifact = wandb.Artifact(name="<置き換え>", type="<置き換え>")
```

上記のコードスニペットの文字列引数を、独自の名前とタイプに置き換えてください。
#### 2. アーティファクトにさらにファイルを追加する

ファイル、ディレクトリ、外部URI参照（Amazon S3など）をアーティファクトのメソッドで追加できます。例えば、1つのテキストファイルを追加するには、[`add_file`](https://docs.wandb.ai/ref/python/artifact#add\_file) メソッドを使用します。

```python
artifact.add_file(local_path="hello_world.txt", name="optional-name")
```
ファイルの追加方法については、[`add_dir`](https://docs.wandb.ai/ref/python/artifact#add\_dir) メソッドを使って複数のファイルを追加することもできます。ファイルの追加方法の詳細については、[アーティファクトの更新](https://docs.wandb.ai/guides/artifacts/update-an-artifact)をご覧ください。

#### 3. アーティファクトをWeights & Biasesサーバーに保存する

最後に、アーティファクトをWeights & Biasesサーバーに保存します。アーティファクトはrunと関連付けられています。そのため、runオブジェクトの[`log_artifact()`](https://docs.wandb.ai/ref/python/run#log\_artifact)メソッドを使ってアーティファクトを保存します。

```python
# W&B Runを作成します。'job-type'を置き換えてください。
run = wandb.init(project="artifacts-example", job_type="job-type")

run.log_artifact(artifact)
```
W&B runの外でアーティファクトをオプションで構築することもできます。詳細は、[外部ファイルのトラッキング](https://docs.wandb.ai/guides/artifacts/track-external-files)を参照してください。

:::caution
`log_artifact`への呼び出しは、パフォーマンスのために非同期で行われます。これにより、ループでアーティファクトをログする際に、予期しない振る舞いが起こることがあります。例えば：

```python
for i in range(10):
    a = wandb.Artifact(
        "race",
        type="dataset",
        metadata={
            "index": i,
        },
    )
    # ... アーティファクトaにファイルを追加 ...
    run.log_artifact(a)
```
アーティファクトのバージョン **v0** は、アーティファクトが任意の順序でログに記録される可能性があるため、メタデータでインデックス0が保証されているわけではありません。
:::

## アーティファクトにファイルを追加する

次のセクションでは、さまざまなファイルタイプや並列実行からアーティファクトを構築する方法をデモンストレーションします。
以下の例では、複数のファイルとディレクトリ構造があるプロジェクトディレクトリを持っていると仮定してください：

```
プロジェクト-ディレクトリ
|-- 画像
|   |-- 猫.png
|   +-- 犬.png
|-- チェックポイント
|   +-- モデル.h5
+-- モデル.h5
```
### シングルファイルの追加

以下のコードスニペットは、ローカルファイルをアーティファクトに追加する方法を示しています。

```python
# シングルファイルの追加
artifact.add_file(local_path="path/file.format")
```
例えば、作業用のローカルディレクトリーに`'file.txt'`という名前のファイルがあるとします。

```python
artifact.add_file("path/file.txt")  # `file.txt`として追加されます
```

アーティファクトには、次のような内容が含まれています:

```
file.txt
```

オプションとして、`name`パラメータにアーティファクト内での希望のパスを指定できます。

```python
artifact.add_file(local_path="path/file.format", name="new/path/file.format")
```
アーティファクトは次のように保存されます:

```
new/path/file.txt
```

| API呼び出し                                              | 結果のアーティファクト  |
| ------------------------------------------------------ | ------------------- |
| `artifact.add_file('model.h5')`                        | model.h5            |
| `artifact.add_file('checkpoints/model.h5')`            | model.h5            |
| `artifact.add_file('model.h5', name='models/mymodel.h5')` | models/mymodel.h5   |
### 複数ファイルを追加する

以下のコードスニペットは、ローカルディレクトリ全体をアーティファクトに追加する方法を示しています:

```python
# ディレクトリを再帰的に追加する
artifact.add_dir(local_path="path/file.format", name="optional-prefix")
```
以下のAPI呼び出しは、以下のアーティファクトの内容を生成します:

| API Call                                    | 結果のアーティファクト                                 |
| ------------------------------------------- | ---------------------------------------------------- |
| `artifact.add_dir('images')`                | <p><code>cat.png</code></p><p><code>dog.png</code></p> |
| `artifact.add_dir('images', name='images')` | `name='images')images/cat.pngimages/dog.png`         |
| `artifact.new_file('hello.txt')`            | `hello.txt`                                          |
### URIリファレンスを追加する

URIがW&Bライブラリが扱い方を知っているスキームを持っている場合、アーティファクトはチェックサムや再現性のための他の情報をトラッキングします。

外部URIリファレンスをアーティファクトに追加するには、[`add_reference`](https://docs.wandb.ai/ref/python/artifact#add\_reference)メソッドを使用します。`'uri'`文字列を自分のURIに置き換えてください。オプションで、名前パラメータにアーティファクト内での所望のパスを渡すことができます。

```python
# URI参照の追加
artifact.add_reference(uri="uri", name="optional-name")
```

アーティファクトは現在以下のURIスキームをサポートしています。

* `http(s)://`: HTTP経由でアクセス可能なファイルへのパス。HTTPサーバーが`ETag`および`Content-Length`レスポンスヘッダーをサポートしている場合、アーティファクトはetagとサイズのメタデータの形式でチェックサムを追跡します。
* `s3://`: S3内のオブジェクトまたはオブジェクトプレフィックスへのパス。アーティファクトは、参照されたオブジェクトのチェックサムとバージョン情報（バケットがオブジェクトのバージョン管理を有効にしている場合）を追跡します。オブジェクトプレフィックスは、プレフィックスの下のオブジェクトを最大10,000オブジェクトまで含めて展開されます。
* `gs://`: GCS内のオブジェクトまたはオブジェクトプレフィックスへのパス。アーティファクトは、参照されたオブジェクトのチェックサムとバージョン情報（バケットがオブジェクトのバージョン管理を有効にしている場合）を追跡します。オブジェクトプレフィックスは、プレフィックスの下のオブジェクトを最大10,000オブジェクトまで含めて展開されます。
以下のAPI呼び出しは、以下のアーティファクトを生成します:

| API呼び出し                                                                      | 結果として得られるアーティファクトの内容                                          |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `artifact.add_reference('s3://my-bucket/model.h5')`                           | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/checkpoints/model.h5')`               | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/model.h5', name='models/mymodel.h5')` | `models/mymodel.h5`                                                  |
| `artifact.add_reference('s3://my-bucket/images')`                             | <p><code>cat.png</code></p><p><code>dog.png</code></p>               |
| `artifact.add_reference('s3://my-bucket/images', name='images')`              | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |
### 並列実行からアーティファクトにファイルを追加する

大規模なデータセットや分散トレーニングの場合、複数の並行実行が1つのアーティファクトに寄与することが必要になることがあります。

```python
import wandb
import time

# rayを使って並列で実行するrunsを起動します
# これはデモ目的です。並列で実行する方法は
# あなたが望むように調整できます。
import ray

ray.init()

artifact_type = "データセット"
artifact_name = "parallel-artifact"
table_name = "distributed_table"
parts_path = "parts"
num_parallel = 5

# 並列で書き込む各バッチは、固有のグループ名を
# 持っている必要があります。
group_name = "writer-group-{}".format(round(time.time()))


@ray.remote
def train(i):
    """
    書き込みジョブです。各ライターは、アーティファクトに1つの画像を追加します。
    """
    with wandb.init(group=group_name) as run:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)

        # wandbテーブルにデータを追加します。 この場合は、例のデータを使用します
        table = wandb.Table(columns=["a", "b", "c"], data=[[i, i * 2, 2**i]])

        # アーティファクト内のフォルダにテーブルを追加します
        artifact.add(table, "{}/table_{}".format(parts_path, i))

        # アーティファクトをアップサートすると、アーティファクトにデータが作成されたり、追加されたりします
        run.upsert_artifact(artifact)


# 並行してrunを実行する
result_ids = [train.remote(i) for i in range(num_parallel)]

# すべてのファイルがアーティファクトに追加されるのを確認するために、
# すべてのライターに結合します。
ray.get(result_ids)

# すべてのライターが終了したら、アーティファクトを
# 完了して準備ができていることを示します。
with wandb.init(group=group_name) as run:
    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    # テーブルのフォルダを指す"PartitionTable"を作成し
    # アーティファクトに追加します。
    artifact.add(wandb.data_types.PartitionedTable(parts_path), table_name)

    # Finish artifact は、アーティファクトの確定を行い、今後このバージョンへの"upserts"を禁止します
    run.finish_artifact(artifact)
```