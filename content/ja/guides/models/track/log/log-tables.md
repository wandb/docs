---
title: テーブルをログする
description: W&B でテーブルをログする
menu:
  default:
    identifier: ja-guides-models-track-log-log-tables
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}
`wandb.Table` を使うと、データをログして W&B 上で可視化・クエリできます。このガイドでは、以下の使い方を学びます:

1. [テーブルの作成]({{< relref path="./log-tables.md#create-tables" lang="ja" >}})
2. [データの追加]({{< relref path="./log-tables.md#add-data" lang="ja" >}})
3. [データの取得]({{< relref path="./log-tables.md#retrieve-data" lang="ja" >}})
4. [テーブルの保存]({{< relref path="./log-tables.md#save-tables" lang="ja" >}})

## テーブルの作成

Table を定義するには、各データ行で見たいカラム（列）を指定します。各行は、例えばトレーニングデータセット内のひとつのアイテム、トレーニング中のあるステップやエポック、モデルがテストアイテムに対して出力した予測、モデルが生成したオブジェクト などが該当します。各カラムには「数値」「テキスト」「ブール値」「画像」「動画」「音声」など固定の型があり、事前に型を指定する必要はありません。カラムごとに名前をつけ、そのカラムインデックスには必ず適切な型のデータのみ渡すようにしましょう。より詳しい例は、[W&B Tables ガイド](https://wandb.ai/stacey/mnist-viz/reports/Guide-to-W-B-Tables--Vmlldzo2NTAzOTk#1.-how-to-log-a-wandb.table) をご覧ください。

`wandb.Table` コンストラクタは以下の2通りで利用可能です。

1. **リスト形式の行:** カラム名とデータの行（リスト）を渡してテーブルを作成、たとえば以下のコードスニペットでは2行3列のテーブルを生成します:

```python
wandb.Table(columns=["a", "b", "c"], data=[["1a", "1b", "1c"], ["2a", "2b", "2c"]])
```

2. **Pandas DataFrame:** `wandb.Table(dataframe=my_df)` で DataFrame をログできます。カラム名は DataFrame から自動的に取得されます。

#### 既存の配列や DataFrame から作成

```python
# モデルが4枚の画像に対して予測を返したと仮定します
# 以下のフィールドを持っています:
# - 画像ID
# - 画像ピクセル情報 (wandb.Image() でラップ)
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

テーブルはミュータブル（可変）です。スクリプトの実行途中で最大20万行までデータを追加できます。テーブルへの追加方法は2通りです。

1. **行の追加:** `table.add_data("3a", "3b", "3c")` のように追加できます。新しい行はリストではなく引数として渡す点に注意しましょう。リスト形式の場合、 `*` を使いリストを引数へ展開できます: `table.add_data(*my_row_list)` 。行内の要素数はテーブルのカラム数と一致している必要があります。
2. **カラムの追加:** `table.add_column(name="col_name", data=col_data)` で追加します。`col_data` はテーブルの現在の行数と同じ長さでなければなりません。`col_data` にはリストや NumPy の NDArray も利用可能です。

### データを段階的に追加

以下のサンプルコードでは、事前に決めたカラムを使って W&B テーブルを作成し、全ラベルの信頼度（スコア）も含めて推論の都度1行ずつ追加していきます。 [run を再開した際のテーブルへのデータ追加はこちら]({{< relref path="#adding-data-to-resumed-runs" lang="ja" >}}) も参照ください。

```python
# ラベルごとの信頼度スコアを含むカラムを定義
columns = ["id", "image", "guess", "truth"]
for digit in range(10):  # 各数字(0-9)の信頼度カラムを追加
    columns.append(f"score_{digit}")

# 定義したカラムでテーブルを初期化
test_table = wandb.Table(columns=columns)

# テストデータセットをイテレートし、1行ずつテーブルに追加
# 各行は画像ID・画像・予測ラベル・正解ラベル・信頼度スコアを含む
for img_id, img in enumerate(mnist_test_data):
    true_label = mnist_test_data_labels[img_id]  # 正解ラベル
    guess_label = my_model.predict(img)  # 予測ラベル
    test_table.add_data(
        img_id, wandb.Image(img), guess_label, true_label
    )  # データ行をテーブルへ追加
```

#### 再開 run でのデータ追加

既存のテーブルを Artifact からロードし、最後のデータ行を取得してメトリクスを追加し、互換性のためテーブルを再初期化後、更新したバージョンを再び W&B にログすることで、再開した run で W&B テーブルを段階的にアップデートできます。

```python
import wandb

# run を初期化
with wandb.init(project="my_project") as run:

    # Artifact から既存のテーブルを読み込み
    best_checkpt_table = run.use_artifact(table_tag).get(table_name)

    # テーブル末尾の行を取得（再開用）
    best_iter, best_metric_max, best_metric_min = best_checkpt_table.data[-1]

    # 必要に応じてメトリクスを更新

    # テーブルへ更新したデータを追加
    best_checkpt_table.add_data(best_iter, best_metric_max, best_metric_min)

    # 互換性のためデータを用いてテーブルを再初期化
    best_checkpt_table = wandb.Table(
        columns=["col1", "col2", "col3"], data=best_checkpt_table.data
    )

    # run の初期化
    run = wandb.init()

    # 更新されたテーブルを W&B にログ
    run.log({table_name: best_checkpt_table})
```

## データ取得

Table にデータを保存したら、カラムごとや行ごとにアクセスできます。

1. **行イテレータ:** Table の row イテレータ（`for ndx, row in table.iterrows(): ...` など）でデータ行を効率的にイテレートできます。
2. **カラム取得:** `table.get_column("col_name")` で任意のカラムデータを取得可能です。引数 `convert_to="numpy"` を指定すると NumPy NDArray のプリミティブ配列に変換できます。カラム内が `wandb.Image` などメディア型の場合も、この方法で生データに直接アクセスできます。

## テーブル保存

例えばモデルの予測結果など、スクリプト内で生成したテーブルを W&B に保存することで、リアルタイムに可視化できます。

### run へのテーブルのログ

`wandb.Run.log()` を使って作成したテーブルを run に保存します。

```python
with wandb.init() as run:
    my_table = wandb.Table(columns=["a", "b"], data=[["1a", "1b"], ["2a", "2b"]])
    run.log({"table_key": my_table})
```

同じキーでテーブルをログするたびに新しいバージョンがバックエンドに保存されます。これにより、複数のトレーニングステップで同じテーブルを記録して、モデル予測の改善を時系列で追跡したり、異なる run 間でテーブルを比較することも可能です（同じキーであれば問題ありません）。20万行までログできます。

{{% alert %}}
20万行以上をログしたい場合は、下記のように上限値を上書きできます。

`wandb.Table.MAX_ARTIFACT_ROWS = X`

ただし、UI でのクエリ速度低下などパフォーマンスが劣化する可能性があります。
{{% /alert %}}

### プログラムからのテーブルアクセス

バックエンドでは Tables は Artifacts として保存されます。特定バージョンの Table をアクセスしたい場合は Artifact API を利用できます。

```python
with wandb.init() as run:
    my_table = run.use_artifact("run-<run-id>-<table-name>:<tag>").get("<table-name>")
```

Artifacts についての詳細は、Developer Guide の [Artifacts Chapter]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を参照ください。

### テーブルの可視化

この方法でログされたテーブルは、Workspace 上の Run Page や Project Page で閲覧できます。詳細は [テーブルの可視化と分析]({{< relref path="/guides/models/tables//visualize-tables.md" lang="ja" >}}) をご覧ください。

## Artifact テーブル

`artifact.add()` を使って、テーブルを run の Workspace ではなく Artifacts セクションにログできます。例えば一度だけ保存し、今後の run で再利用したいデータセットがある場合などに便利です。

```python
with wandb.init(project="my_project") as run:
    # 重要なステップごとに wandb Artifact を作成
    test_predictions = wandb.Artifact("mnist_test_preds", type="predictions")

    # [上記のように predictions 用のデータを作成]
    test_table = wandb.Table(data=data, columns=columns)
    test_predictions.add(test_table, "my_test_key")
    run.log_artifact(test_predictions)
```

画像データを例にした `artifact.add()` の詳しい使い方は [この Colab](https://wandb.me/dsviz-nature-colab) を、Artifacts と Tables を活用して [表データのバージョン管理・重複排除する事例レポート](https://wandb.me/TBV-Dedup) も参照してください。

### Artifact テーブルの join

`wandb.JoinedTable(table_1, table_2, join_key)` を使い、ローカルで作成したテーブルや他 Artifact から取得したテーブルを join（結合）できます。

| 引数      | 説明                                                                                                        |
| --------- | ---------------------------------------------------------------------------------------------------------- |
| table_1  | (str, `wandb.Table`, ArtifactEntry) Artifact 内の `wandb.Table` のパス/テーブルオブジェクト/ArtifactEntry  |
| table_2  | (str, `wandb.Table`, ArtifactEntry) Artifact 内の `wandb.Table` のパス/テーブルオブジェクト/ArtifactEntry  |
| join_key | (str, [str, str]) 結合時のキー                                                                             |

Artifact 内に既にログした2つの Tables を join して新しい Table を作成する場合、Artifact から取り出して join し、その結果をあらたな W&B Table としてアップロードできます。

例として、 `'original_songs'` という元楽曲の Table と、同じ楽曲を合成した `'synth_songs'` テーブルを `"song_id"` で join して、新たなテーブルを作成・アップロードするコード例を示します。

```python
import wandb

with wandb.init(project="my_project") as run:

    # 元楽曲のテーブルを取得
    orig_songs = run.use_artifact("original_songs:latest")
    orig_table = orig_songs.get("original_samples")

    # 合成楽曲のテーブルを取得
    synth_songs = run.use_artifact("synth_songs:latest")
    synth_table = synth_songs.get("synth_samples")

    # "song_id" でテーブルを join
    join_table = wandb.JoinedTable(orig_table, synth_table, "song_id")
    join_at = wandb.Artifact("synth_summary", "analysis")

    # Artifact にテーブルを追加し、W&B へログ
    join_at.add(join_table, "synth_explore")
    run.log_artifact(join_at)
```

異なる Artifact オブジェクトに格納された2つのテーブルを結合する事例は [このチュートリアル](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM) も参考にしてください。