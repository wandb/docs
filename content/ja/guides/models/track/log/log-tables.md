---
title: テーブルをログする
description: W&B でテーブルをログする。
menu:
  default:
    identifier: ja-guides-models-track-log-log-tables
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}
`wandb.Table` を使ってデータをログし、W&B で可視化・クエリできます。このガイドでは次のことを学びます:

1. [Create Tables]({{< relref path="./log-tables.md#create-tables" lang="ja" >}})
2. [Add Data]({{< relref path="./log-tables.md#add-data" lang="ja" >}})
3. [Retrieve Data]({{< relref path="./log-tables.md#retrieve-data" lang="ja" >}})
4. [Save Tables]({{< relref path="./log-tables.md#save-tables" lang="ja" >}})

## Table を作成 {#create-tables}

Table を定義するには、各行のデータに対して表示したい列を指定します。各行はトレーニングデータセットの 1 アイテム、トレーニング中の特定のステップやエポック、テストアイテムに対するモデルの予測、モデルが生成したオブジェクトなどを表せます。各列には固定の型があります（数値、テキスト、ブール値、画像、動画、音声など）。型を事前に指定する必要はありません。各列に名前を付け、その列のインデックスにはその型のデータだけを渡してください。詳細な例は [W&B Tables ガイド](https://wandb.ai/stacey/mnist-viz/reports/Guide-to-W-B-Tables--Vmlldzo2NTAzOTk#1.-how-to-log-a-wandb.table) を参照してください。

`wandb.Table` のコンストラクタは 2 通りの使い方があります:

1. List of Rows: 列名と行データをログします。たとえば、次のコードスニペットは 2 行 3 列の Table を生成します:

```python
wandb.Table(columns=["a", "b", "c"], data=[["1a", "1b", "1c"], ["2a", "2b", "2c"]])
```

2. Pandas DataFrame: `wandb.Table(dataframe=my_df)` で DataFrame をログします。列名は DataFrame から抽出されます.

#### 既存の配列または DataFrame から

```python
# モデルが 4 枚の画像に対する予測を返し、
# 次のフィールドが利用可能だとします:
# - 画像 ID
# - wandb.Image() でラップされた画像データ
# - モデルの予測ラベル
# - 正解ラベル
my_data = [
    [0, wandb.Image("img_0.jpg"), 0, 0],
    [1, wandb.Image("img_1.jpg"), 8, 0],
    [2, wandb.Image("img_2.jpg"), 7, 1],
    [3, wandb.Image("img_3.jpg"), 1, 1],
]

# 対応する列を持つ wandb.Table() を作成
columns = ["id", "image", "prediction", "truth"]
test_table = wandb.Table(data=my_data, columns=columns)
```

## データを追加 {#add-data}

Table は可変です。スクリプトの実行中に最大 200,000 行までデータを追加できます。追加方法は 2 通りあります:

1. 行を追加: `table.add_data("3a", "3b", "3c")`。新しい行はリストとしてではなく引数として渡します。行がリスト形式である場合は、アンパック演算子 `*` を使って位置引数に展開します: `table.add_data(*my_row_list)`。行の要素数は Table の列数と一致している必要があります。
2. 列を追加: `table.add_column(name="col_name", data=col_data)`。`col_data` の長さは Table の現在の行数と同じでなければなりません。`col_data` にはリストや NumPy の NDArray を渡せます。

### データを段階的に追加

以下のコードサンプルは、W&B の Table を段階的に作成・投入する方法を示します。すべての可能なラベルに対する信頼度スコア列も含めて事前に列を定義し、推論時に行ごとにデータを追加します。[Run を再開した際に Table に段階的にデータを追加]({{< relref path="#adding-data-to-resumed-runs" lang="ja" >}}) することもできます。

```python
# 各ラベルの信頼度スコア列を含む Table の列を定義
columns = ["id", "image", "guess", "truth"]
for digit in range(10):  # 各数字 (0-9) の信頼度スコア列を追加
    columns.append(f"score_{digit}")

# 定義した列で Table を初期化
test_table = wandb.Table(columns=columns)

# テストデータセットを反復して、行ごとに Table にデータを追加
# 各行には画像 ID、画像、予測ラベル、真のラベル、信頼度スコアが含まれます
for img_id, img in enumerate(mnist_test_data):
    true_label = mnist_test_data_labels[img_id]  # 正解ラベル
    guess_label = my_model.predict(img)  # 予測ラベル
    test_table.add_data(
        img_id, wandb.Image(img), guess_label, true_label
    )  # 行データを Table に追加
```

#### 再開した Run へのデータ追加 {#adding-data-to-resumed-runs}

既存の Table を Artifact から読み込み、最後の行データを取得してメトリクスを更新し、互換性のために Table を再初期化してから更新版を W&B に再ログすることで、再開した Run でも W&B の Table を段階的に更新できます。

```python
import wandb

# Run を初期化
with wandb.init(project="my_project") as run:

    # 既存の Table を Artifact から読み込む
    best_checkpt_table = run.use_artifact(table_tag).get(table_name)

    # 再開用に Table の最後の行データを取得
    best_iter, best_metric_max, best_metric_min = best_checkpt_table.data[-1]

    # 必要に応じて最良メトリクスを更新

    # 更新データを Table に追加
    best_checkpt_table.add_data(best_iter, best_metric_max, best_metric_min)

    # 互換性を確保するため、更新後のデータで Table を再初期化
    best_checkpt_table = wandb.Table(
        columns=["col1", "col2", "col3"], data=best_checkpt_table.data
    )

    # Run を初期化
    run = wandb.init()

    # 更新した Table を W&B にログ
    run.log({table_name: best_checkpt_table})
```

## データの取得 {#retrieve-data}

データが Table に入ったら、列または行でアクセスできます:

1. Row Iterator: `for ndx, row in table.iterrows(): ...` のように Table の行イテレータを使って、各行を効率的に反復処理できます。
2. 列を取得: `table.get_column("col_name")` で列データを取得できます。便宜的に、`convert_to="numpy"` を渡すとその列をプリミティブの NumPy NDArray に変換できます。列に `wandb.Image` などのメディア型が含まれている場合、基になるデータに直接アクセスするのに有用です。

## Table の保存 {#save-tables}

スクリプト内でデータの Table（例: モデル予測の Table）を生成したら、結果をライブで可視化できるよう W&B に保存します。

### Table を Run にログ

`wandb.Run.log()` を使用して、次のように Table を Run に保存します:

```python
with wandb.init() as run:
    my_table = wandb.Table(columns=["a", "b"], data=[["1a", "1b"], ["2a", "2b"]])
    run.log({"table_key": my_table})
```

同じキーに Table をログするたびに、バックエンドに新しいバージョンの Table が作成・保存されます。つまり、同じキーにログし続ければ、複数のトレーニングステップにわたって同じ Table をログしてモデル予測の改善を追跡したり、異なる Run 間で Table を比較したりできます。最大 200,000 行までログ可能です。

{{% alert %}}
200,000 行を超えてログしたい場合は、次のように上限を上書きできます:

`wandb.Table.MAX_ARTIFACT_ROWS = X`

ただし、これにより UI でのクエリの低速化などパフォーマンス問題が発生する可能性があります。
{{% /alert %}}

### プログラムから Table にアクセス

バックエンドでは、Tables は Artifacts として永続化されています。特定のバージョンにアクセスしたい場合は、Artifact API を使って次のように取得できます:

```python
with wandb.init() as run:
    my_table = run.use_artifact("run-<run-id>-<table-name>:<tag>").get("<table-name>")
```

Artifacts の詳細は、Developer Guide の [Artifacts Chapter]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を参照してください。

### Table を可視化

この方法でログした Table は、Workspace の Run Page と Project Page の両方に表示されます。詳しくは [Visualize and Analyze Tables]({{< relref path="/guides/models/tables//visualize-tables.md" lang="ja" >}}) を参照してください。

## Artifact の Table

`artifact.add()` を使うと、Workspace ではなく Run の Artifacts セクションに Table をログできます。これは、1 回だけログして今後の Run から参照したい Dataset がある場合などに便利です。

```python
with wandb.init(project="my_project") as run:
    # 重要な各ステップごとに wandb Artifact を作成
    test_predictions = wandb.Artifact("mnist_test_preds", type="predictions")

    # [上記のように予測データを作成]
    test_table = wandb.Table(data=data, columns=columns)
    test_predictions.add(test_table, "my_test_key")
    run.log_artifact(test_predictions)
```

画像データを使った `artifact.add()` の詳細な例はこの Colab を、Artifacts と Tables を使って[表形式データのバージョン管理と重複排除](https://wandb.me/TBV-Dedup)を行う例はこの Report を参照してください: [detailed example of artifact.add() with image data](https://wandb.me/dsviz-nature-colab)

### Artifact Table の結合

ローカルで作成した Table、または他の Artifact から取得した Table を、`wandb.JoinedTable(table_1, table_2, join_key)` で結合できます。

| 引数      | 説明                                                                                                              |
| --------- | ----------------------------------------------------------------------------------------------------------------- |
| table_1  | (str, `wandb.Table`, ArtifactEntry) Artifact 内の `wandb.Table` へのパス、Table オブジェクト、または ArtifactEntry |
| table_2  | (str, `wandb.Table`, ArtifactEntry) Artifact 内の `wandb.Table` へのパス、Table オブジェクト、または ArtifactEntry |
| join_key | (str, [str, str]) 結合を実行するキー（複数可）                                                                      |

以前に Artifact コンテキストでログした 2 つの Tables を結合するには、それぞれを Artifact から取得し、結果を新しい Table に結合します。

たとえば、次のコード例では、'original_songs' という元の楽曲の Table と、同じ楽曲の合成版の Table 'synth_songs' を読み込みます。"song_id" で 2 つの Table を結合し、結果の Table を新しい W&B Table としてアップロードします:

```python
import wandb

with wandb.init(project="my_project") as run:

    # 元の楽曲の Table を取得
    orig_songs = run.use_artifact("original_songs:latest")
    orig_table = orig_songs.get("original_samples")

    # 合成楽曲の Table を取得
    synth_songs = run.use_artifact("synth_songs:latest")
    synth_table = synth_songs.get("synth_samples")

    # "song_id" で結合
    join_table = wandb.JoinedTable(orig_table, synth_table, "song_id")
    join_at = wandb.Artifact("synth_summary", "analysis")

    # Table を Artifact に追加して W&B にログ
    join_at.add(join_table, "synth_explore")
    run.log_artifact(join_at)
```

異なる Artifact オブジェクトに保存された 2 つの Table を結合する例は、[このチュートリアル](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) を参照してください。