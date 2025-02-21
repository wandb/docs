---
title: Log tables
description: W&B でテーブルをログします。
menu:
  default:
    identifier: ja-guides-models-track-log-log-tables
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}
`wandb.Table` を使用して、データをログに記録し、Weights & Biases（W&B）で可視化とクエリを実行します。このガイドでは、次の方法を学びます。

1. [テーブルの作成]({{< relref path="./log-tables.md#create-tables" lang="ja" >}})
2. [データの追加]({{< relref path="./log-tables.md#add-data" lang="ja" >}})
3. [データの取得]({{< relref path="./log-tables.md#retrieve-data" lang="ja" >}})
4. [テーブルの保存]({{< relref path="./log-tables.md#save-tables" lang="ja" >}})

## テーブルの作成

Tableを定義するには、データの各行に表示する列を指定します。各行は、トレーニングデータセット内の単一のアイテム、トレーニング中の特定の手順またはエポック、テストアイテムに対するモデルによる予測、モデルによって生成されたオブジェクトなどです。各列には、数値、テキスト、ブール値、画像、ビデオ、オーディオなどの固定タイプがあります。タイプを事前に指定する必要はありません。各列に名前を付け、その列のインデックスにそのタイプのデータのみを渡すようにしてください。詳細な例については、[このレポート](https://wandb.ai/stacey/mnist-viz/reports/Guide-to-W-B-Tables--Vmlldzo2NTAzOTk#1.-how-to-log-a-wandb.table) を参照してください。

`wandb.Table` コンストラクターは、次の2つの方法で使用します。

1. **行のリスト：** 名前付きの列とデータの行を記録します。たとえば、次のコードスニペットは、2つの行と3つの列を持つテーブルを生成します。

```python
wandb.Table(columns=["a", "b", "c"], data=[["1a", "1b", "1c"], ["2a", "2b", "2c"]])
```

2. **Pandas DataFrame：** `wandb.Table(dataframe=my_df)` を使用して DataFrame を記録します。列名は DataFrame から抽出されます。

#### 既存の配列またはデータフレームから

```python
# モデルが4つの画像について予測を返し、
# 次のフィールドが利用可能であると仮定します。
# - 画像 ID
# - wandb.Image() にラップされた画像ピクセル
# - モデルの予測ラベル
# - 正解ラベル
my_data = [
    [0, wandb.Image("img_0.jpg"), 0, 0],
    [1, wandb.Image("img_1.jpg"), 8, 0],
    [2, wandb.Image("img_2.jpg"), 7, 1],
    [3, wandb.Image("img_3.jpg"), 1, 1],
]

# 対応する列を持つ wandb.Table() を作成します。
columns = ["id", "image", "prediction", "truth"]
test_table = wandb.Table(data=my_data, columns=columns)
```

## データの追加

Table は変更可能です。スクリプトの実行中に、テーブルに最大 200,000 行のデータを追加できます。テーブルにデータを追加する方法は2つあります。

1. **行の追加**： `table.add_data("3a", "3b", "3c")` 。新しい行はリストとして表されないことに注意してください。行がリスト形式の場合は、スター表記 `*` を使用して、リストを位置引数に展開します： `table.add_data(*my_row_list)` 。行には、テーブル内の列と同じ数のエントリが含まれている必要があります。
2. **列の追加**： `table.add_column(name="col_name", data=col_data)` 。`col_data` の長さは、テーブルの現在の行数と等しくなければならないことに注意してください。ここで、 `col_data` はリストデータ、または NumPy NDArray にすることができます。

### データを段階的に追加する

このコードサンプルは、W&B table を段階的に作成して入力する方法を示しています。可能なすべてのラベルの信頼性スコアを含む、事前定義された列を持つテーブルを定義し、推論中にデータを1行ずつ追加します。[run を再開するときにテーブルにデータを段階的に追加する]({{< relref path="#adding-data-to-resumed-runs" lang="ja" >}}) こともできます。

```python
# 各ラベルの信頼性スコアを含む、テーブルの列を定義します。
columns = ["id", "image", "guess", "truth"]
for digit in range(10):  # 各数字（0〜9）の信頼性スコア列を追加します
    columns.append(f"score_{digit}")

# 定義された列でテーブルを初期化します
test_table = wandb.Table(columns=columns)

# テストデータセットを反復処理し、データを1行ずつテーブルに追加します
# 各行には、画像ID、画像、予測ラベル、正解ラベル、および信頼性スコアが含まれます。
for img_id, img in enumerate(mnist_test_data):
    true_label = mnist_test_data_labels[img_id]  # 正解ラベル
    guess_label = my_model.predict(img)  # 予測ラベル
    test_table.add_data(
        img_id, wandb.Image(img), guess_label, true_label
    )  # 行データをテーブルに追加します
```

#### 再開された Runs へのデータの追加

Artifacts から既存のテーブルをロードし、データの最後の行を取得して、更新されたメトリクスを追加することにより、再開された Runs で W&B table を段階的に更新できます。次に、互換性のためにテーブルを再初期化し、更新されたバージョンを W&B に記録します。

```python
# Artifacts から既存のテーブルをロードします
best_checkpt_table = wandb.use_artifact(table_tag).get(table_name)

# 再開のためにテーブルからデータの最後の行を取得します
best_iter, best_metric_max, best_metric_min = best_checkpt_table.data[-1]

# 必要に応じて、最適なメトリックを更新します

# 更新されたデータをテーブルに追加します
best_checkpt_table.add_data(best_iter, best_metric_max, best_metric_min)

# 互換性を確保するために、更新されたデータでテーブルを再初期化します
best_checkpt_table = wandb.Table(
    columns=["col1", "col2", "col3"], data=best_checkpt_table.data
)

# 更新されたテーブルを Weights & Biases に記録します
wandb.log({table_name: best_checkpt_table})
```

## データの取得

Table にデータが入ったら、列または行でアクセスします。

1. **行イテレーター**：ユーザーは、`for ndx, row in table.iterrows(): ...` などの Table の行イテレーターを使用して、データの行を効率的に反復処理できます。
2. **列の取得**：ユーザーは、 `table.get_column("col_name")` を使用してデータの列を取得できます。便宜上、ユーザーは `convert_to="numpy"` を渡して、列をプリミティブの NumPy NDArray に変換できます。これは、列に `wandb.Image` などのメディアタイプが含まれており、基になるデータに直接アクセスできる場合に役立ちます。

## テーブルの保存

スクリプトでデータのテーブル（たとえば、モデル予測のテーブル）を生成した後、結果をライブで可視化するために、W&B に保存します。

### Run にテーブルを記録する

`wandb.log()` を使用して、次のようにテーブルを Run に保存します。

```python
run = wandb.init()
my_table = wandb.Table(columns=["a", "b"], data=[["1a", "1b"], ["2a", "2b"]])
run.log({"table_key": my_table})
```

テーブルが同じキーに記録されるたびに、テーブルの新しいバージョンが作成され、バックエンドに保存されます。これは、モデルの予測が時間の経過とともにどのように改善されるかを確認したり、同じキーに記録されている限り、異なる Runs 間でテーブルを比較したりするために、複数のトレーニングステップで同じテーブルを記録できることを意味します。最大200,000行を記録できます。

{{% alert %}}
200,000行を超える行数を記録するには、次のコードで制限を上書きできます。

`wandb.Table.MAX_ARTIFACT_ROWS = X`

ただし、これにより、UI でのクエリの遅延など、パフォーマンスの問題が発生する可能性があります。
{{% /alert %}}

### プログラムでテーブルにアクセスする

バックエンドでは、Table は Artifacts として永続化されます。特定のバージョンにアクセスする場合は、Artifacts API を使用してアクセスできます。

```python
with wandb.init() as run:
    my_table = run.use_artifact("run-<run-id>-<table-name>:<tag>").get("<table-name>")
```

Artifacts の詳細については、開発者ガイドの [Artifacts のチャプター]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を参照してください。

### テーブルの可視化

この方法で記録されたテーブルは、Run ページと Project ページの両方の Workspace に表示されます。詳細については、[テーブルの可視化と分析]({{< relref path="/guides/core/tables/visualize-tables.md" lang="ja" >}}) を参照してください。

## Artifacts テーブル

Workspace の代わりに、Run の Artifacts セクションにテーブルを記録するには、 `artifact.add()` を使用します。これは、一度記録して、将来の Runs で参照するデータセットがある場合に役立ちます。

```python
run = wandb.init(project="my_project")
# 意味のあるステップごとに wandb Artifact を作成します
test_predictions = wandb.Artifact("mnist_test_preds", type="predictions")

# [上記のように予測データを構築します]
test_table = wandb.Table(data=data, columns=columns)
test_predictions.add(test_table, "my_test_key")
run.log_artifact(test_predictions)
```

画像データを使用した `artifact.add()` の[詳細な例については、この Colab](http://wandb.me/dsviz-nature-colab) を参照し、Artifacts と Tables を使用して[表形式のデータをバージョン管理および重複排除する方法の例については、このレポート](http://wandb.me/TBV-Dedup) を参照してください。

### Artifact テーブルの結合

ローカルで構築したテーブル、または他の Artifacts から取得したテーブルを、 `wandb.JoinedTable(table_1, table_2, join_key)` を使用して結合できます。

| 引数       | 説明                                                                                                                       |
| ---------- | ------------------------------------------------------------------------------------------------------------------------- |
| table_1   | (str, `wandb.Table`, ArtifactEntry) Artifacts 内の `wandb.Table` へのパス、テーブルオブジェクト、または ArtifactEntry |
| table_2   | (str, `wandb.Table`, ArtifactEntry) Artifacts 内の `wandb.Table` へのパス、テーブルオブジェクト、または ArtifactEntry |
| join_key | (str, [str, str]) 結合を実行するキー                                                                                      |

Artifact コンテキストで以前に記録した2つの Table を結合するには、Artifact からそれらを取得し、結果を新しい Table に結合します。

たとえば、`'original_songs'` という元の曲の Table と、同じ曲の合成バージョンの別の Table `'synth_songs'` を読み取る方法を示します。次のコード例では、2つのテーブルを `"song_id"` で結合し、結果のテーブルを新しい W&B Table としてアップロードします。

```python
import wandb

run = wandb.init(project="my_project")

# 元の曲のテーブルを取得します
orig_songs = run.use_artifact("original_songs:latest")
orig_table = orig_songs.get("original_samples")

# 合成された曲のテーブルを取得します
synth_songs = run.use_artifact("synth_songs:latest")
synth_table = synth_songs.get("synth_samples")

# テーブルを "song_id" で結合します
join_table = wandb.JoinedTable(orig_table, synth_table, "song_id")
join_at = wandb.Artifact("synth_summary", "analysis")

# Artifact にテーブルを追加して W&B に記録します
join_at.add(join_table, "synth_explore")
run.log_artifact(join_at)
```

異なる Artifact オブジェクトに保存されている以前に保存された2つのテーブルを結合する方法の例については、[このチュートリアル](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) を参照してください。
