---
description: Log tables with W&B.
displayed_sidebar: ja
---

# テーブルのログ

`wandb.Table`を使ってデータをログし、W&Bで可視化・検索を行います。このガイドでは、以下の方法を学びます。

1. [テーブルの作成](./log-tables#create-tables)
2. [データの追加](./log-tables#add-data)
3. [データの取得](./log-tables#retrieve-data)
4. [テーブルの保存](./log-tables#save-tables)

## テーブルの作成

テーブルを定義するには、各行のデータに対して表示したい列を指定します。各行は、トレーニングデータセット内の単一のアイテム、トレーニング中の特定のステップやエポック、テストアイテムでのモデルによる予測、モデルが生成したオブジェクトなどが考えられます。各列には固定のタイプがあります：数値、テキスト、ブール値、画像、ビデオ、オーディオなどです。事前にタイプを指定する必要はありません。各列に名前を付け、その列インデックスにそのタイプのデータを指定してください。より詳細な例については、[このレポート](https://wandb.ai/stacey/mnist-viz/reports/Guide-to-W-B-Tables--Vmlldzo2NTAzOTk#1.-how-to-log-a-wandb.table)を参照してください。

`wandb.Table`コンストラクタを以下の2つの方法で使用します。

1. **列と行のリスト:** 列と行のデータをログします。たとえば、次のコードスニペットでは、2行3列の表が生成されます。

```python
wandb.Table(columns=["a", "b", "c"], 
            data=[["1a", "1b", "1c"], ["2a", "2b", "2c"]])
```

2. **Pandas データフレーム:** `wandb.Table(dataframe=my_df)`を使ってデータフレームをログします。列名はデータフレームから抽出されます。

#### 既存の配列やデータフレームから
```python
# 4つの画像に対してモデルが予測を返したと仮定
# 以下のフィールドが利用可能:
# - 画像ID
# - wandb.Image()でラップされた画像のピクセル
# - モデルの予測ラベル
# - 正解ラベル
my_data = [
  [0, wandb.Image("img_0.jpg"), 0, 0],
  [1, wandb.Image("img_1.jpg"), 8, 0],
  [2, wandb.Image("img_2.jpg"), 7, 1],
  [3, wandb.Image("img_3.jpg"), 1, 1]
]

# 対応する列を持つwandb.Table()を作成
columns=["id", "image", "prediction", "truth"]
test_table = wandb.Table(data=my_data, columns=columns)
```

## データ追加

テーブルは変更可能です。スクリプトが実行されると、最大200,000行までテーブルにデータを追加できます。テーブルにデータを追加する方法は二つあります。

1. **行を追加**: `table.add_data("3a", "3b", "3c")`。新しい行はリストとして表されません。行がリスト形式の場合、アスタリスク記号`*`を使用して、リストを位置引数に展開してください。: `table.add_data(*my_row_list)`。 行は、テーブルの列数と同じ数のエントリを含む必要があります。
2. **列を追加**: `table.add_column(name="col_name", data=col_data)`。 ここで、`col_data` の長さは、テーブルの現在の行数と同じでなければなりません。 ここで、`col_data` はリストデータ、またはNumPy NDArrayです。

#### データを逐次追加

```python


＃上と同じ列を持つテーブルを作成し、
＃すべてのラベルの信頼スコアを追加
columns=["id", "image", "guess", "truth"]
for digit in range(10):
    columns.append("score_" + str(digit))
test_table = wandb.Table(columns=columns)

＃すべての画像で推論を実行し、my_modelが
＃予測ラベルを返し、正解ラベルが利用可能であることを前提とします
for img_id, img in enumerate(mnist_test_data):
    true_label = mnist_test_data_labels[img_id]
    guess_label = my_model.predict(img)
    test_table.add_data(img_id, wandb.Image(img), \
                         guess_label, true_label)
```

## データの取得

データがテーブルに入ったら、列または行でアクセスできます。

1. **行イテレータ**：ユーザーは、`for ndx, row in table.iterrows(): ...`のように、テーブルの行イテレータを使用して、データの行を効率的に反復処理できます。
2. **列の取得**：ユーザーは、`table.get_column("col_name")`を使用してデータの列を取得できます。便宜上、ユーザーは`convert_to="numpy"`を渡して、列をプリミティブのNumPy NDArrayに変換できます。これは、`wandb.Image`などのメディアタイプを含む列がある場合に便利で、直接基本データにアクセスできます。

## テーブルの保存

スクリプト内でデータのテーブルを生成した後、たとえばモデル予測のテーブルなど、W&Bに保存してライブで結果を可視化できます。

### ランにテーブルをログする

`wandb.log()`を使用して、ランにテーブルを保存します。
```python
run = wandb.init()
my_table = wandb.Table(columns=["a", "b"], data=[["1a", "1b"], ["2a", "2b"]])
run.log({"table_key": my_table})
```

同じキーに対してテーブルが記録されるたびに、バックエンドで新しいバージョンのテーブルが作成され、保存されます。これにより、同じテーブルを複数のトレーニングステップにわたってログして、モデルの予測がどのように時間とともに向上しているかを確認したり、同じキーにログされた異なるrunsのテーブルを比較したりできます。最大200,000行までログできます。

:::info
200,000行以上のログには、以下の方法で制限をオーバーライドできます。

`wandb.Table.MAX_ROWS = X`

ただし、これによりUIで遅いクエリなどのパフォーマンス問題が発生する可能性があります。
:::

### プログラムでテーブルにアクセスする

バックエンドでは、テーブルはアーティファクトとして永続化されています。特定のバージョンにアクセスしたい場合は、アーティファクトAPIを使ってアクセスできます：

```python
with wandb.init() as run:
   my_table = run.use_artifact("run-<run-id>-<table-name>:<tag>").get("<table-name>")
```

アーティファクトについての詳細は、開発者ガイドの[Artifacts Chapter](../../artifacts/intro.md) を参照してください。

## テーブルの可視化

この方法でログされた任意のテーブルは、ワークスペース内のRunページおよびProjectページの両方に表示されます。詳細については、[テーブルの可視化と分析](../../tables/visualize-tables.md)を参照してください。
## 上級編：アーティファクトテーブル

`artifact.add()`を使用して、ワークスペースの代わりにrunのArtifactsセクションにテーブルを記録します。これは、一度ログを記録して、将来のrunで参照したいデータセットがある場合に役立ちます。

```python
run = wandb.init(project="my_project")
# それぞれの意味のあるステップでwandb Artifactを作成する
test_predictions = wandb.Artifact("mnist_test_preds", 
                                  type="predictions")
                                  
# [上記のように予測データを組み立てます]
test_table = wandb.Table(data=data, columns=columns)
test_predictions.add(test_table, "my_test_key")
run.log_artifact(test_predictions)
```

このColabでは、[画像データを用いたartifact.add()の詳細な例](http://wandb.me/dsviz-nature-colab) と、ArtifactsとTablesを使用した[バージョン管理と重複データの削除](http://wandb.me/TBV-Dedup)の方法の例が見られるレポートを参照してください。

### アーティファクトテーブルの結合

ローカルで構築したテーブルや他のアーティファクトから取得したテーブルを、`wandb.JoinedTable(table_1, table_2, join_key)` を使って結合することができます。

| 引数      | 説明                                                                                                          |
| --------- | ------------------------------------------------------------------------------------------------------------- |
| table_1  | (str, `wandb.Table`, ArtifactEntry) アーティファクト内の `wandb.Table` へのパス、テーブルオブジェクト、または ArtifactEntry |
| table_2  | (str, `wandb.Table`, ArtifactEntry) アーティファクト内の `wandb.Table` へのパス、テーブルオブジェクト、または ArtifactEntry |
| join_key | (str, [str, str]) 結合を実行するキーまたはキー                                                                   |
以前にログした2つのテーブルをアーティファクトコンテキストで結合するには、アーティファクトからそれらを取得し、結果を新しいテーブルに結合します。

例として、`'original_songs'`という名前のオリジナル曲のテーブルと、`'synth_songs'`という同じ曲の合成バージョンのテーブルを1つ読み取る方法を示しています。次のコード例では、`"song_id"`で2つのテーブルを結合し、結果のテーブルを新しいW&Bテーブルとしてアップロードします。

```python
run = wandb.init(project="my_project")

# オリジナル曲のテーブルを取得
orig_songs = run.use_artifact('original_songs:latest')
orig_table = orig_songs.get("original_samples")

# 合成された曲のテーブルを取得
synth_songs = run.use_artifact('synth_songs:latest') 
synth_table = synth_songs.get("synth_samples")

# "song_id"でテーブルを結合
join_table = wandb.JoinedTable(orig_table, synth_table, "song_id")
join_at = wandb.Artifact("synth_summary", "analysis")

# テーブルをアーティファクトに追加してW&Bにログ
join_at.add(join_table, "synth_explore")
run.log_artifact(join_at)
```

2つの異なるアーティファクトオブジェクトに格納された以前に保存されたテーブルを組み合わせる方法については、この[Colabノートブック](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)を参照してください。