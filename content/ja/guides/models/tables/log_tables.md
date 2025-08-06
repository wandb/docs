---
title: テーブルをログする
weight: 2
---

W&B Tables を使って表形式のデータを可視化・ログできます。W&B Table は、各列がひとつのデータ型で構成された 2 次元のデータグリッドです。各行は、W&B の [run]({{< relref "/guides/models/track/runs/" >}}) に記録された 1 つ以上のデータポイントを表します。W&B Tables は、プリミティブ型や数値型だけでなく、ネストされたリストや辞書、リッチメディア型にも対応しています。

W&B Table は W&B 独自の [データ型]({{< relref "/ref/python/sdk/data-types/" >}}) であり、[artifact]({{< relref "/guides/core/artifacts/" >}}) オブジェクトとしてログされます。

W&B Python SDK を使って[テーブルオブジェクトを作成・ログ]({{< relref "#create-and-log-a-new-table" >}})できます。テーブルオブジェクトを作成する際は、列情報・データ、そして[モード]({{< relref "#table-logging-modes" >}})を指定します。モードにより、表の記録・更新方法が決まり、ML実験中での扱いが変わります。

{{% alert %}}
`INCREMENTAL` モードは W&B Server v0.70.0 以降でサポートされています。
{{% /alert %}}

## テーブルの作成とログ

1. `wandb.init()` で新しい run を初期化します。
2. [`wandb.Table`]({{< relref "/ref/python/sdk/data-types/table" >}}) クラスで Table オブジェクトを作成します。`columns` と `data` パラメータで列とデータをそれぞれ指定します。オプションの `log_mode` パラメータは、`IMMUTABLE`（デフォルト）、`MUTABLE`、`INCREMENTAL` の3つのモードから選択できます。詳細は次の[テーブルのロギングモード]({{< relref "#logging-modes" >}})をご覧ください。
3. 作成したテーブルを `run.log()` で W&B にログします。

以下は 2 列（`a` と `b`）、2 行（`["a1", "b1"]` と `["a2", "b2"]`）のテーブルを作成・ログする例です。

```python
import wandb

# 新しい run を開始
with wandb.init(project="table-demo") as run:

    # 2 列 2 行のデータでテーブルオブジェクトを作成
    my_table = wandb.Table(
        columns=["a", "b"],
        data=[["a1", "b1"], ["a2", "b2"]],
        log_mode="IMMUTABLE"
        )

    # テーブルを W&B にログ
    run.log({"Table Name": my_table})
```

## ロギングモード

[`wandb.Table`]({{< relref "/ref/python/sdk/data-types/table" >}}) の `log_mode` パラメータは、テーブルの記録・更新方法を決定します。`log_mode` には `IMMUTABLE`、`MUTABLE`、`INCREMENTAL` の3つがあり、テーブルの編集や表示、W&B アプリ上での見え方に違いが出ます。

以下は各ロギングモードの概要とユースケース、特徴です。

| モード  | 定義 | ユースケース  | メリット  |
| ----- | ---------- | ---------- | ----------|
| `IMMUTABLE`   | 一度 W&B に記録されたテーブルは変更できません。 |- run 終了時に生成された表データの保存・後分析用  | - ラン終了時にログした際に最小限の負荷<br>- UIで全行を表示 |
| `MUTABLE`     | ログ後も既存テーブルを新しい内容で上書きできます。 | - 既存テーブルへの列・行追加<br>- 新情報による結果の強化    | - テーブル変更履歴をキャプチャ<br>- UIで全行を表示         |
| `INCREMENTAL` | 実験の進行に応じて新しい行をバッチで追加できます。 | - バッチ単位での行追加<br> - 長時間の学習<br>- 大規模データセット<br>- 継続的な結果監視 | - トレーニング中にUI上で更新を確認<br>- インクリメントごとの確認が可能   |

次のセクションでは各モードに対応したコード例と使用上の注意を紹介します。

### MUTABLE モード

`MUTABLE` モードでは、既存のテーブルが新しい内容で丸ごと上書きされます。既存表に列や行を追記したい場合や、非反復方式で更新したい場合に便利です。UI 上では新規追加した列や行も含めてすべて表示されます。

{{% alert %}}
`MUTABLE` モードでは、ログするたびにテーブルが新しい内容で置き換えられます。大規模テーブルの場合、高コストかつ処理が遅くなる可能性があります。
{{% /alert %}}

以下は、`MUTABLE` モードでテーブルを作成し、ログし、新しい列を追加する例です。テーブルオブジェクトは最初のデータ、信頼度スコア、最終予測値と 3 回記録されます。

{{% alert %}}
例では `load_eval_data()` でデータ取得、`model.predict()` で予測を行う仮の関数を使っていますので、ご自身のデータ取得・予測関数に置き換えてください。
{{% /alert %}}

```python
import wandb
import numpy as np

with wandb.init(project="mutable-table-demo") as run:

    # MUTABLEロギングモードのテーブル作成
    table = wandb.Table(columns=["input", "label", "prediction"],
                        log_mode="MUTABLE")

    # データの読み込みと予測
    inputs, labels = load_eval_data() # 仮の関数
    raw_preds = model.predict(inputs) # 仮の関数

    for inp, label, pred in zip(inputs, labels, raw_preds):
        table.add_data(inp, label, pred)

    # ステップ1: 初期データのログ 
    run.log({"eval_table": table})

    # ステップ2: 信頼度スコア（例: softmax 最大値）追加
    confidences = np.max(raw_preds, axis=1)
    table.add_column("confidence", confidences)
    run.log({"eval_table": table})

    # ステップ3: 後処理予測値の追加
    # 例: スレッショルドやスムージングした出力
    post_preds = (confidences > 0.7).astype(int)
    table.add_column("final_prediction", post_preds)
    run.log({"eval_table": table})
```

繰り返し行のみを新しく追加したい（列は追加しない）場合は、[`INCREMENTAL` モード]({{< relref "#INCREMENTAL-mode" >}}) の利用も検討できます。

### INCREMENTAL モード

`INCREMENTAL` モードでは、実験中にバッチごとに新しい行をテーブルに記録します。長時間ジョブの監視や、大規模テーブルへの効率よい記録に最適です。UI では新規追加された行がその都度更新表示され、run 全体の終了を待つことなく最新データを閲覧可能です。また、インクリメント単位で履歴をさかのぼって表示できます。

{{% alert %}}
W&B App の run workspace では、最大 100 インクリメントまで表示されます。100 回以上ログした場合は直近 100 件のみが表示されます。
{{% /alert %}}

以下は `INCREMENTAL` モードでテーブルを作成し、訓練ループごとに新しい行を追加・ログする例です。テーブルは各 step ごとに記録されます。

{{% alert %}}
以下はデータ取得 `get_training_batch()`、モデル訓練 `train_model_on_batch()`、予測 `predict_on_batch()` を仮の関数として使用していますので、ご自身の実装に置き換えてください。
{{% /alert %}}

```python
import wandb

with wandb.init(project="incremental-table-demo") as run:

    # INCREMENTALロギングモードでテーブル作成
    table = wandb.Table(columns=["step", "input", "label", "prediction"],
                        log_mode="INCREMENTAL")

    # トレーニングループ
    for step in range(get_num_batches()): # 仮の関数
        # バッチデータの読み込み
        inputs, labels = get_training_batch(step) # 仮の関数

        # 訓練と予測
        train_model_on_batch(inputs, labels) # 仮の関数
        predictions = predict_on_batch(inputs) # 仮の関数

        # バッチデータをテーブルに追加
        for input_item, label, prediction in zip(inputs, labels, predictions):
            table.add_data(step, input_item, label, prediction)

        # テーブルをインクリメンタルにログ
        run.log({"training_table": table}, step=step)
```

インクリメンタルロギングは（`log_mode=MUTABLE` で毎回新しいテーブルをログする場合に比べて）計算コストを抑えることができます。ただし、インクリメント回数が多い場合、W&B App で全行が表示されないことがあります。run 中にテーブルの更新確認と、run 終了後の全データ分析を両立したい場合は、`INCREMENTAL` モードのテーブルと `IMMUTABLE` モードのテーブルを 2 つ併用する方法もあります。

下記は `INCREMENTAL` と `IMMUTABLE` のロギングを組み合わせた例です。

```python
import wandb

with wandb.init(project="combined-logging-example") as run:

    # トレーニング中の効率的な更新用インクリメンタルテーブル
    incr_table = wandb.Table(columns=["step", "input", "prediction", "label"],
                            log_mode="INCREMENTAL")

    # トレーニングループ
    for step in range(get_num_batches()):
        # バッチ処理
        inputs, labels = get_training_batch(step)
        predictions = model.predict(inputs)

        # インクリメンタルテーブルにデータ追加
        for inp, pred, label in zip(inputs, predictions, labels):
            incr_table.add_data(step, inp, pred, label)

        # インクリメンタルアップデートをログ (最終テーブルと区別するため -incr 接尾辞)
        run.log({"table-incr": incr_table}, step=step)

    # トレーニング完了時、全データから完全なIMMUTABLEテーブル作成
    # デフォルトのIMMUTABLEモードで全データセットを保存
    final_table = wandb.Table(columns=incr_table.columns, data=incr_table.data, log_mode="IMMUTABLE")
    run.log({"table": final_table})
```

この例では、トレーニング中は `incr_table` を `log_mode="INCREMENTAL"` で継続的にログし、run 終了時にすべてのデータを `IMMUTABLE` テーブル（`final_table`）として保存しています。こうすることで run 中の進捗を逐次モニタリングしつつ、最終的な全データをW&B App で詳細分析できます。

## 実例

### MUTABLE を使った評価結果の強化

```python
import wandb
import numpy as np

with wandb.init(project="mutable-logging") as run:

    # ステップ1: 初期予測をログ
    table = wandb.Table(columns=["input", "label", "prediction"], log_mode="MUTABLE")
    inputs, labels = load_eval_data()
    raw_preds = model.predict(inputs)

    for inp, label, pred in zip(inputs, labels, raw_preds):
        table.add_data(inp, label, pred)

    run.log({"eval_table": table})  # 生予測のログ

    # ステップ2: 信頼度スコア（例: softmax 最大値）追加
    confidences = np.max(raw_preds, axis=1)
    table.add_column("confidence", confidences)
    run.log({"eval_table": table})  # 信頼度の追加

    # ステップ3: 後処理した予測値を追加
    # 例: 閾値処理やスムージング出力
    post_preds = (confidences > 0.7).astype(int)
    table.add_column("final_prediction", post_preds)
    run.log({"eval_table": table})
```

### INCREMENTAL テーブルで run の再開

途中から run を再開し、インクリメンタルテーブルへのログを継続できます。

```python
# run の新規または再開
resumed_run = wandb.init(project="resume-incremental", id="your-run-id", resume="must")

# インクリメンタルテーブルの作成（前回のデータ投入は不要）
# Table artifact へのインクリメント追加を継続
table = wandb.Table(columns=["step", "metric"], log_mode="INCREMENTAL")

# ログ再開
for step in range(resume_step, final_step):
    metric = compute_metric(step)
    table.add_data(step, metric)
    resumed_run.log({"metrics": table}, step=step)

resumed_run.finish()
```

{{% alert %}}
`wandb.Run.define_metric("<table_key>", summary="none")` や `wandb.Run.define_metric("*", summary="none")` でインクリメンタルテーブル用キーのサマリーを切ると新しいテーブルとしてインクリメントが記録されます。
{{% /alert %}}


### INCREMENTAL バッチトレーニングでの活用

```python

with wandb.init(project="batch-training-incremental") as run:

    # インクリメンタルテーブルを作成
    table = wandb.Table(columns=["step", "input", "label", "prediction"], log_mode="INCREMENTAL")

    # シミュレートされたトレーニングループ
    for step in range(get_num_batches()):
        # バッチデータをロード
        inputs, labels = get_training_batch(step)

        # モデルの学習
        train_model_on_batch(inputs, labels)

        # モデル推論
        predictions = predict_on_batch(inputs)

        # テーブルへデータ追加
        for input_item, label, prediction in zip(inputs, labels, predictions):
            table.add_data(step, input_item, label, prediction)

        # 現在のテーブル状態をインクリメンタルにログ
        run.log({"training_table": table}, step=step)
```