---
title: テーブルをログする
description: W&B でテーブルをログする。
menu:
  default:
    identifier: log-tables
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}
`wandb.Table` を使ってデータをログし、W&B で可視化・クエリできます。このガイドでは、以下の内容を学びます。

1. [テーブルの作成]({{< relref "./log-tables.md#create-tables" >}})
2. [データの追加]({{< relref "./log-tables.md#add-data" >}})
3. [データの取得]({{< relref "./log-tables.md#retrieve-data" >}})
4. [テーブルの保存]({{< relref "./log-tables.md#save-tables" >}})

## テーブルの作成

Table を定義するには、各データ行に表示したいカラムを指定します。各行はトレーニングデータセットの 1 サンプルや、トレーニング中の特定のステップまたはエポック、モデルがテストサンプルに対して行った予測、あるいはモデルが生成したオブジェクトなどに対応します。各カラムには固定の型（数値、テキスト、ブール値、画像、動画、音声など）があり、型を事前に指定する必要はありません。カラムに名前を付け、そのカラムインデックスに同じ型のデータのみを渡してください。より詳しい例は、[W&B Tables ガイド](https://wandb.ai/stacey/mnist-viz/reports/Guide-to-W-B-Tables--Vmlldzo2NTAzOTk#1.-how-to-log-a-wandb.table) をご覧ください。

`wandb.Table` のコンストラクタは次の 2 通りの方法で使えます。

1. **リストによる各行の指定:** 名前付きカラムとデータ行をログします。次のコード例は、2 行 3 カラムのテーブルを生成します。

```python
wandb.Table(columns=["a", "b", "c"], data=[["1a", "1b", "1c"], ["2a", "2b", "2c"]])
```

2. **Pandas DataFrame:** `wandb.Table(dataframe=my_df)` で DataFrame をログできます。カラム名は DataFrame から自動で抽出されます。

#### 既存の配列や DataFrame から作成

```python
# モデルが4つの画像に対して予測を出したと仮定します
# 以下の項目が利用可能です:
# - 画像のID
# - 画像のピクセルデータ (wandb.Image() でラップ)
# - モデルの予測ラベル
# - 正解ラベル
my_data = [
    [0, wandb.Image("img_0.jpg"), 0, 0],
    [1, wandb.Image("img_1.jpg"), 8, 0],
    [2, wandb.Image("img_2.jpg"), 7, 1],
    [3, wandb.Image("img_3.jpg"), 1, 1],
]

# 対応するカラムで wandb.Table() を作成
columns = ["id", "image", "prediction", "truth"]
test_table = wandb.Table(data=my_data, columns=columns)
```

## データの追加

Tables はミュータブル（変更可能）です。スクリプトの実行中、最大 200,000 行までデータを追加できます。データを追加する方法は 2 つあります。

1. **行の追加:** `table.add_data("3a", "3b", "3c")` のように新しい行を追加します。行がリスト形式の場合は `*` を使ってリストを展開し、位置引数として渡します: `table.add_data(*my_row_list)`。行の要素数はカラム数と同じである必要があります。
2. **カラムの追加:** `table.add_column(name="col_name", data=col_data)` でカラムを追加可能です。`col_data` の長さは、テーブルの現在の行数と同じでなければなりません。`col_data` にはリストや NumPy NDArray を指定できます。

### データを段階的に追加する

このサンプルコードは、W&B のテーブルを事前に定義したカラムで作成し、推論時にデータを 1 行ずつ追加していく様子を示しています。各ラベルの信頼度スコアを含めたカラムを定義し、推論時に逐次データを追加します。[Resumed run でテーブルに追加する方法]({{< relref "#adding-data-to-resumed-runs" >}}) も参考にしてください。

```python
# 各ラベルの信頼度を含めたカラムを定義
columns = ["id", "image", "guess", "truth"]
for digit in range(10):  # ラベルごとの信頼度カラムを追加 (0〜9)
    columns.append(f"score_{digit}")

# 定義したカラムでテーブルを初期化
test_table = wandb.Table(columns=columns)

# テストデータセットをイテレートして 1 行ずつデータを追加
# 各行には画像ID、画像、予測ラベル、正解ラベル、信頼度スコアが含まれる
for img_id, img in enumerate(mnist_test_data):
    true_label = mnist_test_data_labels[img_id]  # 正解ラベル
    guess_label = my_model.predict(img)  # 予測ラベル
    test_table.add_data(
        img_id, wandb.Image(img), guess_label, true_label
    )  # 行データをテーブルに追加
```

#### Resumed run へのデータ追加

Resumed run では、既存のテーブルを artifact からロードし、最後の行を取得してメトリクスを更新し、互換性のためにテーブルを再初期化してから W&B に再ログできます。

```python
import wandb

# Run を初期化
with wandb.init(project="my_project") as run:

    # 既存のテーブルを artifact からロード
    best_checkpt_table = run.use_artifact(table_tag).get(table_name)

    # 再開用にテーブルの最後の行データを取得
    best_iter, best_metric_max, best_metric_min = best_checkpt_table.data[-1]

    # 必要に応じて best メトリクスを更新

    # 更新したデータをテーブルに追加
    best_checkpt_table.add_data(best_iter, best_metric_max, best_metric_min)

    # 互換性のため、更新されたデータでテーブルを再初期化
    best_checkpt_table = wandb.Table(
        columns=["col1", "col2", "col3"], data=best_checkpt_table.data
    )

    # Run を初期化
    run = wandb.init()

    # 更新済みテーブルを W&B にログ
    run.log({table_name: best_checkpt_table})
```

## データの取得

Table にデータが入ったら、カラムまたは行ごとにアクセスできます。

1. **行イテレーター:** Table の row イテレータを使って、`for ndx, row in table.iterrows(): ...` のように効率的に行を巡ることができます。
2. **カラム取得:** `table.get_column("col_name")` でカラムデータを取得可能です。`convert_to="numpy"` を指定すると NumPy NDArray に変換できます。カラムが `wandb.Image` のようなメディア型の場合、元データへ直接アクセスできます。

## テーブルの保存

例えばモデルの予測テーブルなど、スクリプトでデータのテーブルを作成したら、それを W&B に保存して、結果をリアルタイムで可視化しましょう。

### テーブルを run にログする

`wandb.Run.log()` を使ってテーブルを run に保存します。例：

```python
with wandb.init() as run:
    my_table = wandb.Table(columns=["a", "b"], data=[["1a", "1b"], ["2a", "2b"]])
    run.log({"table_key": my_table})
```

同じキーでテーブルをログするたびに、新しいバージョンが作成され、バックエンドに保存されます。これにより、複数のトレーニングステップで同じテーブルをログして、モデル予測の変化を可視化したり、異なる run 間で比較したりできます（同じキーでログされている場合）。最大 200,000 行までログ可能です。

{{% alert %}}
20 万行を超えてログしたい場合は、次のように制限を上書きできます。

`wandb.Table.MAX_ARTIFACT_ROWS = X`

ただし、その場合 UI でクエリが遅くなるなど、パフォーマンスの問題が発生する可能性があります。
{{% /alert %}}

### テーブルへプログラムからアクセスする

バックエンドでは、Tables は Artifacts として保存されます。特定バージョンへアクセスしたい場合は、artifact API を利用してください。

```python
with wandb.init() as run:
    my_table = run.use_artifact("run-<run-id>-<table-name>:<tag>").get("<table-name>")
```

Artifacts の詳細は [Artifacts Chapter]({{< relref "/guides/core/artifacts/" >}}) をご確認ください。

### テーブルの可視化

この方法でログしたテーブルは、Workspace の Run ページ・Project ページ両方に表示されます。詳細は [テーブルの可視化と分析]({{< relref "/guides/models/tables//visualize-tables.md" >}}) をご覧ください。

## アーティファクトテーブル

`artifact.add()` を使うことで、テーブルを run の Artifacts セクションにログできます（Workspace ではなく）。これは 1 度だけデータセットを記録し、今後の run で参照したい場合などに便利です。

```python
with wandb.init(project="my_project") as run:
    # 意味のある各ステップごとに Artifact を作成
    test_predictions = wandb.Artifact("mnist_test_preds", type="predictions")

    # [上述と同様に予測データを構築]
    test_table = wandb.Table(data=data, columns=columns)
    test_predictions.add(test_table, "my_test_key")
    run.log_artifact(test_predictions)
```

画像データを使った artifact.add() の詳しい例は [こちらの Colab](https://wandb.me/dsviz-nature-colab) を、Artifacts と Tables を活用した表データのバージョン管理や重複排除例は [こちらの Report](https://wandb.me/TBV-Dedup) を参照してください。

### アーティファクトテーブルの結合

`wandb.JoinedTable(table_1, table_2, join_key)` を使い、ローカルで構築したテーブルや他の artifact から取得したテーブル同士を結合できます。

| 引数      | 説明                                                                                          |
| --------- | --------------------------------------------------------------------------------------------- |
| table_1  | (str, `wandb.Table`, ArtifactEntry) artifact 内の wandb.Table へのパス、テーブル本体、または ArtifactEntry |
| table_2  | (str, `wandb.Table`, ArtifactEntry) artifact 内の wandb.Table へのパス、テーブル本体、または ArtifactEntry |
| join_key | (str, [str, str]) 結合に使うキーまたはキー群                                                     |

過去に artifact にログした 2 つの Tables を結合する場合、artifact から取得し、新しい Table に結合できます。

例えば、次のコード例では 'original_songs' というオリジナル楽曲 Table と、同じ楽曲の合成版の Table 'synth_songs' を、"song_id" で結合し、新たな W&B テーブルとしてアップロードしています。

```python
import wandb

with wandb.init(project="my_project") as run:

    # オリジナル楽曲テーブルを取得
    orig_songs = run.use_artifact("original_songs:latest")
    orig_table = orig_songs.get("original_samples")

    # 合成楽曲テーブルを取得
    synth_songs = run.use_artifact("synth_songs:latest")
    synth_table = synth_songs.get("synth_samples")

    # "song_id" でテーブルを結合
    join_table = wandb.JoinedTable(orig_table, synth_table, "song_id")
    join_at = wandb.Artifact("synth_summary", "analysis")

    # テーブルを artifact に追加し、W&B にログ
    join_at.add(join_table, "synth_explore")
    run.log_artifact(join_at)
```

異なる Artifact オブジェクトに格納された 2 つのテーブルを組み合わせる具体例は、[このチュートリアル](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) をご覧ください。