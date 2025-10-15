---
title: ログ テーブル
description: W&B でテーブルをログします。
menu:
  default:
    identifier: ja-guides-models-track-log-log-tables
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}
`wandb.Table` を使って、データをログに記録し W&B で視覚化・クエリできるようにします。このガイドでは次のことを学びます：

1. [テーブルを作成する]({{< relref path="./log-tables.md#create-tables" lang="ja" >}})
2. [データを追加する]({{< relref path="./log-tables.md#add-data" lang="ja" >}})
3. [データを取得する]({{< relref path="./log-tables.md#retrieve-data" lang="ja" >}})
4. [テーブルを保存する]({{< relref path="./log-tables.md#save-tables" lang="ja" >}})

## テーブルを作成する

Table（テーブル）を定義するには、各データ行に表示したい列を指定します。各行はトレーニングデータセットの単一の項目、トレーニング中の特定のステップやエポック、テスト項目でのモデルの予測、モデルが生成したオブジェクトなどです。各列には固定の型があり、数値、テキスト、ブール値、画像、ビデオ、オーディオなどがあります。あらかじめ型を指定する必要はありません。各列に名前を付け、その型のデータのみをその列のインデックスに渡してください。より詳細な例については、[このレポート](https://wandb.ai/stacey/mnist-viz/reports/Guide-to-W-B-Tables--Vmlldzo2NTAzOTk#1.-how-to-log-a-wandb.table) を参照してください。

`wandb.Table` コンストラクタを次の2つの方法のいずれかで使用します：

1. **行のリスト:** 名前付きの列とデータの行をログに記録します。例えば、次のコードスニペットは 2 行 3 列のテーブルを生成します：

```python
wandb.Table(columns=["a", "b", "c"], data=[["1a", "1b", "1c"], ["2a", "2b", "2c"]])
```

2. **Pandas DataFrame:** `wandb.Table(dataframe=my_df)` を使用して DataFrame をログに記録します。列の名前は DataFrame から抽出されます。

#### 既存の配列またはデータフレームから

```python
# モデルが4つの画像で予測を返したと仮定します
# 次のフィールドが利用可能です:
# - 画像ID
# - 画像ピクセル（wandb.Image() でラップ）
# - モデルの予測ラベル
# - 正解ラベル
my_data = [
    [0, wandb.Image("img_0.jpg"), 0, 0],
    [1, wandb.Image("img_1.jpg"), 8, 0],
    [2, wandb.Image("img_2.jpg"), 7, 1],
    [3, wandb.Image("img_3.jpg"), 1, 1],
]

# 対応する列で wandb.Table() を作成
columns = ["id", "image", "prediction", "truth"]
test_table = wandb.Table(data=my_data, columns=columns)
```

## データを追加する

Tables は可変です。スクリプトが実行中に最大 200,000 行までテーブルにデータを追加できます。テーブルにデータを追加する方法は2つあります：

1. **行を追加する**: `table.add_data("3a", "3b", "3c")`。新しい行はリストとして表現されないことに注意してください。行がリスト形式の場合は `*` を使ってリストを位置引数に展開します: `table.add_data(*my_row_list)`。行にはテーブルの列数と同じ数のエントリが含まれている必要があります。
2. **列を追加する**: `table.add_column(name="col_name", data=col_data)`。`col_data` の長さは現在のテーブルの行数と同じである必要があります。ここで `col_data` はリストデータや NumPy NDArray でも構いません。

### データを段階的に追加する

このコードサンプルは、次第に W&B テーブルを作成し、データを追加する方法を示しています。信頼度スコアを含む事前定義された列でテーブルを定義し、推論中に行ごとにデータを追加します。また、[run を再開するときにテーブルにデータを段階的に追加]({{< relref path="#adding-data-to-resumed-runs" lang="ja" >}})することもできます。

```python
# 各ラベルの信頼度スコアを含むテーブルの列を定義
columns = ["id", "image", "guess", "truth"]
for digit in range(10):  # 各数字 (0-9) に対する信頼度スコア列を追加
    columns.append(f"score_{digit}")

# 定義された列でテーブルを初期化
test_table = wandb.Table(columns=columns)

# テストデータセットを通過し、データを行ごとにテーブルに追加
# 各行は画像 ID、画像、予測されたラベル、正解ラベル、信頼度スコアを含みます
for img_id, img in enumerate(mnist_test_data):
    true_label = mnist_test_data_labels[img_id]  # 正解ラベル
    guess_label = my_model.predict(img)  # 予測ラベル
    test_table.add_data(
        img_id, wandb.Image(img), guess_label, true_label
    )  # テーブルに行データを追加
```

#### Run を再開した際にデータを追加

再開した Run において、既存のテーブルをアーティファクトから読み込み、最後のデータ行を取得して、更新されたメトリクスを追加することで W&B テーブルを段階的に更新できます。次に、互換性を保つためにテーブルを再初期化し、更新されたバージョンを W&B に再度ログに記録します。

```python
# アーティファクトから既存のテーブルを読み込む
best_checkpt_table = wandb.use_artifact(table_tag).get(table_name)

# 再開のためにテーブルの最後のデータ行を取得
best_iter, best_metric_max, best_metric_min = best_checkpt_table.data[-1]

# 必要に応じて最高のメトリクスを更新

# テーブルに更新されたデータを追加
best_checkpt_table.add_data(best_iter, best_metric_max, best_metric_min)

# 更新されたデータでテーブルを再初期化し、互換性を確保
best_checkpt_table = wandb.Table(
    columns=["col1", "col2", "col3"], data=best_checkpt_table.data
)

# 更新されたテーブルを Weights & Biases にログする
wandb.log({table_name: best_checkpt_table})
```

## データを取得する

データが Table にあるとき、列または行ごとにアクセスできます：

1. **行イテレータ**: ユーザーは Table の行イテレータを利用して、`for ndx, row in table.iterrows(): ...` のようにデータの行を効率的に反復処理できます。
2. **列を取得する**: ユーザーは `table.get_column("col_name")` を使用してデータの列を取得できます。`convert_to="numpy"` を渡すと、列を NumPy のプリミティブ NDArray に変換できます。これは、列に `wandb.Image` などのメディアタイプが含まれている場合に、基になるデータに直接アクセスするのに便利です。

## テーブルを保存する

スクリプトでモデルの予測のテーブルなどのデータを生成した後、それを W&B に保存して結果をリアルタイムで視覚化します。

### Run にテーブルをログする

`wandb.log()` を使用してテーブルを Run に保存します：

```python
run = wandb.init()
my_table = wandb.Table(columns=["a", "b"], data=[["1a", "1b"], ["2a", "2b"]])
run.log({"table_key": my_table})
```

同じキーにテーブルがログに記録されるたびに、新しいバージョンのテーブルが作成され、バックエンドに保存されます。これにより、複数のトレーニングステップにわたって同じテーブルをログに記録し、モデルの予測がどのように向上するかを確認したり、異なる Run 間でテーブルを比較したりすることができます。同じテーブルに最大 200,000 行までログに記録できます。

{{% alert %}}
200,000 行以上をログに記録するには、以下のように制限をオーバーライドできます：

`wandb.Table.MAX_ARTIFACT_ROWS = X`

ただし、これにより UI でのクエリの速度低下などのパフォーマンス問題が発生する可能性があります。
{{% /alert %}}

### プログラムによるテーブルへのアクセス

バックエンドでは、Tables は Artifacts として保存されています。特定のバージョンにアクセスする場合は、artifact API を使用して行うことができます：

```python
with wandb.init() as run:
    my_table = run.use_artifact("run-<run-id>-<table-name>:<tag>").get("<table-name>")
```

Artifacts の詳細については、デベロッパーガイドの [Artifacts チャプター]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を参照してください。

### テーブルを視覚化する

この方法でログに記録されたテーブルは、Run ページと Project ページの両方でワークスペースに表示されます。詳細については、[テーブルの視覚化と分析]({{< relref path="/guides/models/tables//visualize-tables.md" lang="ja" >}}) を参照してください。

## アーティファクトテーブル

`artifact.add()` を使用して、テーブルをワークスペースの代わりに Run の Artifacts セクションにログします。これは、データセットを1回ログに記録し、今後の Run のために参照したい場合に役立ちます。

```python
run = wandb.init(project="my_project")
# 各重要なステップのために wandb Artifact を作成
test_predictions = wandb.Artifact("mnist_test_preds", type="predictions")

# [上記のように予測データを構築]
test_table = wandb.Table(data=data, columns=columns)
test_predictions.add(test_table, "my_test_key")
run.log_artifact(test_predictions)
```

画像データを使用した `artifact.add()` の詳細な例については、この Colab を参照してください: [画像データを使った artifact.add() の詳細な例](http://wandb.me/dsviz-nature-colab) また [Artifacts と Tables を使ったバージョン管理と重複排除データの例](http://wandb.me/TBV-Dedup) に関してはこのレポートを参照してください。

### アーティファクトテーブルを結合する

`wandb.JoinedTable(table_1, table_2, join_key)` を使用して、ローカルに構築したテーブルや他のアーティファクトから取得したテーブルを結合できます。

| 引数      | 説明                                                                                                             |
| --------- | ----------------------------------------------------------------------------------------------------------------- |
| table_1  | (str, `wandb.Table`, ArtifactEntry) アーティファクト内の `wandb.Table` へのパス、テーブルオブジェクト、または ArtifactEntry |
| table_2  | (str, `wandb.Table`, ArtifactEntry) アーティファクト内の `wandb.Table` へのパス、テーブルオブジェクト、または ArtifactEntry |
| join_key | (str, [str, str]) 結合を行うキーまたはキーのリスト                                                                |

以前にアーティファクトコンテキストでログに記録した2つのテーブルを結合するには、アーティファクトからそれらを取得し、新しいテーブルに結合した結果を格納します。この例では、`'original_songs'` という名前のオリジナルの曲のテーブルと `'synth_songs'` という名前の同じ曲の合成バージョンのテーブルを結合する方法を示しています。以下のコード例は 2 つのテーブルを `"song_id"` で結合し、結果のテーブルを新しい W&B テーブルとしてアップロードします：

```python
import wandb

run = wandb.init(project="my_project")

# オリジナルの曲のテーブルを取得
orig_songs = run.use_artifact("original_songs:latest")
orig_table = orig_songs.get("original_samples")

# 合成曲のテーブルを取得
synth_songs = run.use_artifact("synth_songs:latest")
synth_table = synth_songs.get("synth_samples")

# "song_id" でテーブルを結合
join_table = wandb.JoinedTable(orig_table, synth_table, "song_id")
join_at = wandb.Artifact("synth_summary", "analysis")

# テーブルをアーティファクトに追加し W&B にログする
join_at.add(join_table, "synth_explore")
run.log_artifact(join_at)
```

このチュートリアルを[読む](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)と、異なるアーティファクトオブジェクトに保存された 2 つのテーブルを組み合わせる方法の例が示されています。