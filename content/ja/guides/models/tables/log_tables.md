---
title: テーブルをログする
menu:
  default:
    identifier: ja-guides-models-tables-log_tables
weight: 2
---

W&B Tables を使って、表形式データの可視化とログができます。W&B Table は、各列が単一のデータ型を持つ二次元グリッドで、各行は W&B の [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) にログされた 1 件以上のデータポイントを表します。W&B Tables はプリミティブや数値型だけでなく、ネストされたリスト、辞書、リッチメディア型にも対応しています。

W&B Table は W&B 内で利用できる特別な [data type]({{< relref path="/ref/python/sdk/data-types/" lang="ja" >}}) で、[artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) オブジェクトとしてログされます。

W&B Python SDK を利用して [テーブルオブジェクトの作成とログ]({{< relref path="#create-and-log-a-new-table" lang="ja" >}}) を行います。テーブルオブジェクト作成時には、列名やデータ、さらに [mode]({{< relref path="#table-logging-modes" lang="ja" >}}) を指定します。mode によって、テーブルのログや更新の仕方（ML実験中の動き）が決まります。

{{% alert %}}
`INCREMENTAL` モードは W&B Server v0.70.0 以降でサポートされています。
{{% /alert %}}

## テーブルの作成とログ

1. `wandb.init()` で新しい run を初期化します。
2. [`wandb.Table`]({{< relref path="/ref/python/sdk/data-types/table" lang="ja" >}}) クラスで Table オブジェクトを作成します。列情報は `columns`、初期データは `data` パラメータで指定します。オプションの `log_mode` パラメータには、`IMMUTABLE`（デフォルト）、`MUTABLE`、`INCREMENTAL` のいずれかを設定できます。詳細は次セクションの [Table Logging Modes]({{< relref path="#logging-modes" lang="ja" >}}) を参照ください。
3. `run.log()` を使って W&B にテーブルをログします。

下記は、2列 (`a`, `b`)・2行（`["a1", "b1"]` と `["a2", "b2"]`）のテーブルを作成しログする例です。

```python
import wandb

# 新しい run を開始
with wandb.init(project="table-demo") as run:

    # 2列・2行の Table オブジェクトを作成
    my_table = wandb.Table(
        columns=["a", "b"],
        data=[["a1", "b1"], ["a2", "b2"]],
        log_mode="IMMUTABLE"
        )

    # テーブルを W&B にログ
    run.log({"Table Name": my_table})
```

## ログモード

[`wandb.Table`]({{< relref path="/ref/python/sdk/data-types/table" lang="ja" >}}) の `log_mode` パラメータによって、ML実験中のテーブルのログ方法や更新のされ方が決まります。`log_mode` には `IMMUTABLE`, `MUTABLE`, `INCREMENTAL` の3つの引数が設定できます。各モードによるログの挙動・編集可否や、W&B App での表示のされ方が異なります。

以下は３つのモードの特徴および主なユースケースのまとめです。

| Mode  | 定義 | ユースケース  | メリット  |
| ----- | ---- | ------------- | ----------|
| `IMMUTABLE`   | 一度 W&B にテーブルをログすると、その後の編集はできません。 |- run 終了後に生成された表データを保存し、分析を行う場合           | - 終了時のログでは低オーバーヘッド<br>- UI ですべての行が表示可能 |
| `MUTABLE`     | テーブルを W&B にログ後、新しいテーブルで上書きができます | - 既存テーブルへの列・行追加<br>- 新たな情報で結果を充実させる場合 | - テーブルの変更内容を保存可能<br>- UI ですべての行が表示可能     |
| `INCREMENTAL` | ML実験中に新しい行のバッチを追加できます              | - バッチ毎に行を追加<br> - 長時間実行のトレーニング<br>- 大規模データバッチ処理<br>- 実験中の結果の監視 | - トレーニング中に UI で更新状況を確認できる<br>- インクリメント毎に遡って確認可能 |

次のセクションでは、それぞれのモードごとの具体的なコード例・利用時の注意点を解説します。

### MUTABLE モード

`MUTABLE` モードでは、既存テーブルを新しいテーブルで置き換えることで更新されます。イテレーティブでない形で既存テーブルに新しい列・行を追加したいときに便利です。UI 上では、初回ログ後に追加した列・行も含めてすべて表示されます。

{{% alert %}}
`MUTABLE` モードでは、ログの度にテーブルオブジェクト全体が置き換わります。大きなテーブルでは計算コストが高く、ログに時間を要することがあります。
{{% /alert %}}

次は、`MUTABLE` モードで作成・ログし、その後新しい列を追加する例です。初回データ・信頼度スコア・最終予測値と3回テーブルをログします。

{{% alert %}}
この例では、データのロード用 `load_eval_data()` と、予測用 `model.predict()` はダミー関数です。ご自身のデータロード・予測処理に置き換えてください。
{{% /alert %}}

```python
import wandb
import numpy as np

with wandb.init(project="mutable-table-demo") as run:

    # MUTABLE ログモードでテーブル作成
    table = wandb.Table(columns=["input", "label", "prediction"],
                        log_mode="MUTABLE")

    # データをロードして予測を計算
    inputs, labels = load_eval_data() # ダミー関数
    raw_preds = model.predict(inputs) # ダミー関数

    for inp, label, pred in zip(inputs, labels, raw_preds):
        table.add_data(inp, label, pred)

    # ステップ1: 初期データをログ
    run.log({"eval_table": table})  # テーブルをログ

    # ステップ2: 信頼度スコアを追加（例: softmax 最大値）
    confidences = np.max(raw_preds, axis=1)
    table.add_column("confidence", confidences)
    run.log({"eval_table": table})  # 信頼度をログ

    # ステップ3: 最終予測値を追加
    # （例: 阈値処理やスムージングなど後処理）
    post_preds = (confidences > 0.7).astype(int)
    table.add_column("final_prediction", post_preds)
    run.log({"eval_table": table})  # 列追加をログ
```

バッチごとに新しい「行」だけをインクリメンタルに追加していきたい場合（列追加なし）は、[`INCREMENTAL` モード]({{< relref path="#INCREMENTAL-mode" lang="ja" >}}) の利用が便利です。

### INCREMENTAL モード

INCREMENTAL モードでは、ML実験中にバッチ単位でテーブルに行を追加していきます。長大なテーブルを何度も上書きログするのが非効率な場合や、長時間ジョブを監視しながら進捗データを確認したいときに最適です。UI 上も最新の行データが順次追加されていくので、実験終了を待たずに途中経過を追えます。また、インクリメントごとに一時点の状態を振り返ることも可能です。

{{% alert %}}
W&B App の run workspace ではインクリメントは最大 100 回まで表示されます。100 回以上ログした場合は直近 100 回分のみが workspace 上で確認できます。
{{% /alert %}}

次は、INCREMENTAL モードでテーブルを作成し、トレーニングステップごとに新たな行を追加しながらログする例です。

{{% alert %}}
この例でもデータ取得の `get_training_batch()`、学習用 `train_model_on_batch()`、予測用 `predict_on_batch()` はダミー関数です。ご自身のロジックに置き換えてください。
{{% /alert %}}

```python
import wandb

with wandb.init(project="incremental-table-demo") as run:

    # INCREMENTAL ログモードのテーブル作成
    table = wandb.Table(columns=["step", "input", "label", "prediction"],
                        log_mode="INCREMENTAL")

    # トレーニングループ
    for step in range(get_num_batches()): # ダミー関数
        # バッチデータの取得
        inputs, labels = get_training_batch(step) # ダミー関数

        # モデル学習と予測
        train_model_on_batch(inputs, labels) # ダミー関数
        predictions = predict_on_batch(inputs) # ダミー関数

        # バッチのデータをテーブルに追加
        for input_item, label, prediction in zip(inputs, labels, predictions):
            table.add_data(step, input_item, label, prediction)

        # テーブルの状態をインクリメンタルにログ
        run.log({"training_table": table}, step=step)
```

インクリメンタルログは、毎回新規テーブルを作ってログする場合（`log_mode=MUTABLE`）に比べて計算負荷が低くなります。ただし、非常に多くのインクリメントを記録した場合は、W&B App ですべての行が表示されない場合もあります。  
実験の進行中にデータを随時更新＆表示したいかつ、確実に全データを保存・分析したい場合は、INCREMENTAL テーブルと IMMUTABLE テーブルの2つを使い分けるのがおすすめです。

次は `INCREMENTAL` と `IMMUTABLE` ログモードの両方を使う例です。

```python
import wandb

with wandb.init(project="combined-logging-example") as run:

    # トレーニング中の効率更新用に incremental テーブル作成
    incr_table = wandb.Table(columns=["step", "input", "prediction", "label"],
                            log_mode="INCREMENTAL")

    # トレーニングループ
    for step in range(get_num_batches()):
        # バッチ処理
        inputs, labels = get_training_batch(step)
        predictions = model.predict(inputs)

        # incremental テーブルにデータ追加
        for inp, pred, label in zip(inputs, predictions, labels):
            incr_table.add_data(step, inp, pred, label)

        # インクリメンタルな更新（-incr というキーで区別推奨）
        run.log({"table-incr": incr_table}, step=step)

    # トレーニング完了時、全データを使い immutable テーブルを作成
    # デフォルト（IMMUTABLE）でデータセット全体を保存
    final_table = wandb.Table(columns=incr_table.columns, data=incr_table.data, log_mode="IMMUTABLE")
    run.log({"table": final_table})
```

この例では、`incr_table` をトレーニング中にインクリメンタル（`log_mode="INCREMENTAL"`）でログします。こうすることでリアルタイムにテーブル更新と確認ができます。トレーニング終了後には、incremental テーブル中の全データから immutable テーブル（`final_table`）を作成・保存します。immutable テーブルは全データの保存・さらなる分析や W&B App 上での全行表示に使えます。

## 例

### MUTABLE で評価結果を充実させる

```python
import wandb
import numpy as np

with wandb.init(project="mutable-logging") as run:

    # ステップ1: 初期予測値のログ
    table = wandb.Table(columns=["input", "label", "prediction"], log_mode="MUTABLE")
    inputs, labels = load_eval_data()
    raw_preds = model.predict(inputs)

    for inp, label, pred in zip(inputs, labels, raw_preds):
        table.add_data(inp, label, pred)

    run.log({"eval_table": table})  # 予測値をログ

    # ステップ2: 信頼度スコア（max softmax等）追加
    confidences = np.max(raw_preds, axis=1)
    table.add_column("confidence", confidences)
    run.log({"eval_table": table})  # 信頼度を追加

    # ステップ3: 後処理した予測値を追加
    # （閾値処理やスムージングなど）
    post_preds = (confidences > 0.7).astype(int)
    table.add_column("final_prediction", post_preds)
    run.log({"eval_table": table})
```

### INCREMENTAL テーブルで run を再開

run 再開時にも incremental テーブルに続けてログできます。

```python
# run の新規開始または再開
resumed_run = wandb.init(project="resume-incremental", id="your-run-id", resume="must")

# incremental テーブル作成（過去のデータから初期化は不要）
# インクリメントは Table artifact に追加されていきます
table = wandb.Table(columns=["step", "metric"], log_mode="INCREMENTAL")

# 続きからログ
for step in range(resume_step, final_step):
    metric = compute_metric(step)
    table.add_data(step, metric)
    resumed_run.log({"metrics": table}, step=step)

resumed_run.finish()
```

{{% alert %}}
incremental テーブルに使用しているキーでサマリー集計を無効化（例: `wandb.Run.define_metric("<table_key>", summary="none")` または `wandb.Run.define_metric("*", summary="none")`）すると、インクリメントは新しいテーブルとしてログされます。
{{% /alert %}}


### INCREMENTAL バッチトレーニング

```python

with wandb.init(project="batch-training-incremental") as run:

    # incremental テーブル作成
    table = wandb.Table(columns=["step", "input", "label", "prediction"], log_mode="INCREMENTAL")

    # サンプルのトレーニングループ
    for step in range(get_num_batches()):
        # バッチデータ取得
        inputs, labels = get_training_batch(step)

        # このバッチでモデルを学習
        train_model_on_batch(inputs, labels)

        # モデル推論
        predictions = predict_on_batch(inputs)

        # テーブルにデータを追加
        for input_item, label, prediction in zip(inputs, labels, predictions):
            table.add_data(step, input_item, label, prediction)

        # 現在のテーブル状態をインクリメンタルにログ
        run.log({"training_table": table}, step=step)
```