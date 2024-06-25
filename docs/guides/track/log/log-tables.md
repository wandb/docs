---
description: W&Bでテーブルをログする。
displayed_sidebar: default
---


# テーブルのログ

`wandb.Table` を使って、W&Bでデータを視覚化してクエリするためのログを作成します。このガイドでは以下のことを学びます:

1. [テーブルの作成](./log-tables.md#create-tables)
2. [データの追加](./log-tables.md#add-data)
3. [データの取得](./log-tables.md#retrieve-data)
4. [テーブルの保存](./log-tables.md#save-tables)

## テーブルの作成

テーブルを定義するには、各データ行に表示したい列を指定します。各行には、トレーニングデータセットの単一アイテム、トレーニング中の特定のステップやエポック、テストアイテムに対するモデルの予測、モデルによって生成されたオブジェクトなどが含まれます。各列には固定のタイプ（数値、テキスト、ブール値、画像、動画、音声など）がありますが、事前にタイプを指定する必要はありません。各列に名前を付け、対応するタイプのデータをその列のインデックスに渡すようにします。より詳細な例については、[このレポート](https://wandb.ai/stacey/mnist-viz/reports/Guide-to-W-B-Tables--Vmlldzo2NTAzOTk#1.-how-to-log-a-wandb.table)を参照してください。

`wandb.Table` コンストラクタは以下の2つの方法で使用できます:

1. **行のリスト**: 名前付き列とデータの行をログします。例えば、以下のコードスニペットは2行3列のテーブルを生成します:

```python
wandb.Table(columns=["a", "b", "c"], data=[["1a", "1b", "1c"], ["2a", "2b", "2c"]])
```

2. **Pandas DataFrame**: DataFrameを`wandb.Table(dataframe=my_df)` を使ってログします。列名はDataFrameから抽出されます。

#### 既存の配列やデータフレームから

```python
# モデルが4つの画像に対して予測を返したと仮定します
# 以下のフィールドが利用可能です:
# - 画像ID
# - wandb.Image() でラップされた画像ピクセル
# - モデルの予測ラベル
# - 正解ラベル
my_data = [
    [0, wandb.Image("img_0.jpg"), 0, 0],
    [1, wandb.Image("img_1.jpg"), 8, 0],
    [2, wandb.Image("img_2.jpg"), 7, 1],
    [3, wandb.Image("img_3.jpg"), 1, 1],
]

# 対応する列を持つ wandb.Table() を作成します
columns = ["id", "image", "prediction", "truth"]
test_table = wandb.Table(data=my_data, columns=columns)
```

## データの追加

テーブルは可変です。スクリプトの実行中に、最大200,000行までデータを追加できます。テーブルにデータを追加する方法は2つあります:

1. **行の追加**: `table.add_data("3a", "3b", "3c")`。新しい行はリストとして表現しません。行がリスト形式の場合、スター表記 `*` を使用してリストを位置引数に展開します: `table.add_data(*my_row_list)`。行には、テーブルの列数と同じ数のエントリが含まれている必要があります。
2. **列の追加**: `table.add_column(name="col_name", data=col_data)`。`col_data` の長さは現在の行数と等しい必要があります。ここで、`col_data` はリストデータ、またはNumPyのNDArrayである必要があります。

#### データを段階的に追加

```python
# 上述と同じ列を持つテーブルを作成し、
# すべてのラベルの信頼スコアを含めます
columns = ["id", "image", "guess", "truth"]
for digit in range(10):
    columns.append("score_" + str(digit))
test_table = wandb.Table(columns=columns)

# すべての画像に対して推論を実行します。ここで仮定されるのは、
# my_model が予測ラベルを返し、正解ラベルが利用可能であること
for img_id, img in enumerate(mnist_test_data):
    true_label = mnist_test_data_labels[img_id]
    guess_label = my_model.predict(img)
    test_table.add_data(img_id, wandb.Image(img), guess_label, true_label)
```

## データの取得

一度データがテーブルに入ると、列または行ごとにアクセスできます:

1. **行のイテレータ**: Users はテーブルの行イテレータを使用して `for ndx, row in table.iterrows(): ...` のように、データの行を効率的にイテレートできます。
2. **列の取得**: Users は `table.get_column("col_name")` を使用して列データを取得できます。便利なことに、`convert_to="numpy"` を指定して列をNumPyのNDArrayに変換することができます。これにより、`wandb.Image` などのメディアタイプを含む列の基礎データに直接アクセスできるようになります。

## テーブルの保存

スクリプトでデータのテーブル（例えばモデル予測のテーブル）を生成した後、W&Bに保存して結果をライブで視覚化することができます。

### テーブルをrunにログする

`wandb.log()` を使ってテーブルをrunに保存します。以下のように:

```python
run = wandb.init()
my_table = wandb.Table(columns=["a", "b"], data=[["1a", "1b"], ["2a", "2b"]])
run.log({"table_key": my_table})
```

同じキーにテーブルをログするたびに、新しいバージョンのテーブルがバックエンドに作成され保存されます。そのため、複数のトレーニングステップにわたって同じテーブルをログすることで、時間の経過とともにモデル予測の改善を確認したり、異なるruns間でテーブルを比較したりできます。最大で200,000行までログできます。

:::info
200,000行以上をログするには、以下のコマンドで制限をオーバーライドできます:

`wandb.Table.MAX_ARTIFACTS_ROWS = X`

ただし、これによりUIでのクエリ速度低下などのパフォーマンス問題が発生する可能性があります。
:::

### プログラムからテーブルにアクセスする

バックエンドで、テーブルはArtifactsとして保存されます。特定のバージョンにアクセスしたい場合は、Artifact APIを使用してアクセスできます:

```python
with wandb.init() as run:
    my_table = run.use_artifact("run-<run-id>-<table-name>:<tag>").get("<table-name>")
```

Artifactsの詳細については、[Artifacts チャプター](../../artifacts/intro.md)を参照してください。

## テーブルの視覚化

この方法でログされたテーブルはすべて、RunページとProjectページの両方でWorkspaceに表示されます。詳細については、[テーブルの視覚化と分析](../../tables/visualize-tables.md)を参照してください。

## 応用: アーティファクトテーブル

`artifact.add()` を使用して、テーブルをrunのワークスペースのArtfactsセクションにログします。これは、データセットを1回ログし、その後のrunで参照したい場合に便利です。

```python
run = wandb.init(project="my_project")
# 意味のある各ステップのためにwandb Artifactを作成します
test_predictions = wandb.Artifact("mnist_test_preds", type="predictions")

# 予測データを上記のように構築します
test_table = wandb.Table(data=data, columns=columns)
test_predictions.add(test_table, "my_test_key")
run.log_artifact(test_predictions)
```

この[Colabノートブック](http://wandb.me/dsviz-nature-colab) で、画像データとともに artifact.add() を使用する詳細な例を参照し、ArtifactsとTablesを利用して[バージョン管理と表データの重複除去](http://wandb.me/TBV-Dedup) を行う方法を解説したレポートを参照してください。

### アーティファクトテーブルの結合

ローカルで構築したテーブルや他のアーティファクトから取得したテーブルを、`wandb.JoinedTable(table_1, table_2, join_key)` を使用して結合できます。

| Args      | 説明                                                                                                        |
| --------- | ------------------------------------------------------------------------------------------------------------------ |
| table_1  | (str, `wandb.Table`, ArtifactEntry) アーティファクト内の `wandb.Table`へのパス、テーブルオブジェクト、またはArtifactEntry |
| table_2  | (str, `wandb.Table`, ArtifactEntry) アーティファクト内の `wandb.Table`へのパス、テーブルオブジェクト、またはArtifactEntry |
| join_key | (str, [str, str]) 結合を行うキーまたはキーのリスト                                                        |

以前にアーティファクトコンテキストにログした2つのテーブルを結合するには、アーティファクトからそれらを取得し、新しいテーブルに結合結果を挿入します。

例えば、オリジナルの曲のテーブル `'original_songs'` と、その曲の合成バージョンのテーブル `'synth_songs'` を読み取り、以下のコードで2つのテーブルを `"song_id"` で結合し、結果のテーブルを新しいW&Bテーブルとしてアップロードします:

```python
import wandb

run = wandb.init(project="my_project")

# オリジナル曲のテーブルを取得
orig_songs = run.use_artifact("original_songs:latest")
orig_table = orig_songs.get("original_samples")

# 合成曲のテーブルを取得
synth_songs = run.use_artifact("synth_songs:latest")
synth_table = synth_songs.get("synth_samples")

# テーブルを "song_id" で結合
join_table = wandb.JoinedTable(orig_table, synth_table, "song_id")
join_at = wandb.Artifact("synth_summary", "analysis")

# テーブルをアーティファクトに追加し、W&Bにログする
join_at.add(join_table, "synth_explore")
run.log_artifact(join_at)
```

以前に保存された異なるアーティファクトオブジェクトに保存された2つのテーブルを結合する例については、この[Colabノートブック](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)を参照してください。