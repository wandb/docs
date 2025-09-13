---
title: テーブルをログする
menu:
  default:
    identifier: ja-guides-models-tables-log_tables
weight: 2
---

W&B Tables で表形式データを可視化し、ログに記録します。W&B Table は、各列が単一の型を持つ 2 次元のデータ グリッドです。各行は、W&B の [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) にログされた 1 つ以上のデータ ポイントを表します。W&B Tables はプリミティブ型や数値型に加え、入れ子のリスト、辞書、リッチメディア型もサポートします。

W&B Table は W&B における特別な [データ型]({{< relref path="/ref/python/sdk/data-types/" lang="ja" >}}) で、[Artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) オブジェクトとしてログされます。

W&B Python SDK を使って[テーブル オブジェクトを作成してログします]({{< relref path="#create-and-log-a-new-table" lang="ja" >}})。テーブル オブジェクトを作成するときは、テーブルの列とデータ、そして [モード]({{< relref path="#table-logging-modes" lang="ja" >}}) を指定します。モードは、機械学習の実験中にテーブルをどのようにログ・更新するかを決めます。

{{% alert %}}
`INCREMENTAL` モードは W&B Server v0.70.0 以降でサポートされています。
{{% /alert %}}

## テーブルを作成してログする

1. `wandb.init()` で新しい run を初期化します。 
2. [`wandb.Table`]({{< relref path="/ref/python/sdk/data-types/table" lang="ja" >}}) クラスで Table オブジェクトを作成します。`columns` と `data` パラメータで、それぞれテーブルの列とデータを指定します。オプションの `log_mode` パラメータは、`IMMUTABLE`（デフォルト）、`MUTABLE`、`INCREMENTAL` の 3 つのモードのいずれかに設定することを推奨します。詳しくは次のセクションの [テーブルのログ モード]({{< relref path="#logging-modes" lang="ja" >}}) を参照してください。
3. `run.log()` でテーブルを W&B にログします。

次の例は、列 `a` と `b` を持つテーブルを作成し、`["a1", "b1"]` と `["a2", "b2"]` の 2 行をログする方法を示します。

```python
import wandb

# 新しい run を開始
with wandb.init(project="table-demo") as run:

    # 2 列・2 行のデータを持つ Table オブジェクトを作成
    my_table = wandb.Table(
        columns=["a", "b"],
        data=[["a1", "b1"], ["a2", "b2"]],
        log_mode="IMMUTABLE"
        )

    # テーブルを W&B にログする
    run.log({"Table Name": my_table})
```

## ログ モード

[`wandb.Table`]({{< relref path="/ref/python/sdk/data-types/table" lang="ja" >}}) の `log_mode` パラメータは、機械学習の実験中にテーブルをどのようにログし更新するかを決めます。`log_mode` は `IMMUTABLE`、`MUTABLE`、`INCREMENTAL` の 3 つの引数を受け付けます。各モードによって、テーブルのログ方法、変更可否、W&B App での表示が変わります。

以下では 3 つのログ モードの概要と違い、代表的なユースケースを示します。

| Mode  | 定義 | ユースケース | 利点 |
| ----- | ---------- | ---------- | ----------|
| `IMMUTABLE`   | 一度 W&B にログしたテーブルは変更できません。 |- run 終了時に生成された表データを保存し、後で分析する | - run 終了時にまとめてログしてもオーバーヘッドが最小<br>- UI で全行がレンダリングされる |
| `MUTABLE`     | W&B にログした後でも、新しいテーブルで既存のテーブルを上書きできます。 | - 既存のテーブルに列や行を追加する<br>- 新しい情報で結果を充実させる | - テーブルの変更を取り込める<br>- UI で全行がレンダリングされる |
| `INCREMENTAL` | 実験の進行中に、新しい行をバッチで追加します。 | - 行をバッチで追加する<br> - 長時間のトレーニング ジョブ<br>- 大規模データセットをバッチでプロセッシング<br>- 進行中の結果をモニタリング | - トレーニング中に UI で更新を確認できる<br>- インクリメントを段階的に追える |

以降では各モードのコードスニペット例と、使い分けの考慮点を示します。

### MUTABLE モード

`MUTABLE` モードは、既存のテーブルを新しいテーブルで置き換えて更新します。逐次的ではないプロセスで、既存のテーブルに新しい列や行を追加したい場合に便利です。UI では、初回のログ後に追加された新しい列や行も含め、すべての行と列がレンダリングされます。

{{% alert %}}
`MUTABLE` モードでは、テーブルをログするたびにテーブル オブジェクトが置き換わります。新しいテーブルで上書きする処理は計算コストが高く、大きなテーブルでは遅くなる可能性があります。
{{% /alert %}}

次の例では、`MUTABLE` モードでテーブルを作成してログし、その後に新しい列を追加します。テーブル オブジェクトは、初期データ、信頼度スコア、最終予測の 3 回ログされます。

{{% alert %}}
以下の例では、データを読み込むプレースホルダー関数 `load_eval_data()` と、予測を行うプレースホルダー関数 `model.predict()` を使用しています。ご自身のデータ読み込み関数と予測関数に置き換えてください。
{{% /alert %}}

```python
import wandb
import numpy as np

with wandb.init(project="mutable-table-demo") as run:

    # MUTABLE ログ モードで Table オブジェクトを作成
    table = wandb.Table(columns=["input", "label", "prediction"],
                        log_mode="MUTABLE")

    # データを読み込み、予測を作成
    inputs, labels = load_eval_data() # プレースホルダー関数
    raw_preds = model.predict(inputs) # プレースホルダー関数

    for inp, label, pred in zip(inputs, labels, raw_preds):
        table.add_data(inp, label, pred)

    # ステップ 1: 初期データをログ 
    run.log({"eval_table": table})  # 最初のテーブルをログ

    # ステップ 2: 信頼度スコアを追加（例: ソフトマックスの最大値）
    confidences = np.max(raw_preds, axis=1)
    table.add_column("confidence", confidences)
    run.log({"eval_table": table})  # 信頼度情報を追加

    # ステップ 3: 後処理済みの予測を追加
    # （例: 閾値処理や平滑化した出力）
    post_preds = (confidences > 0.7).astype(int)
    table.add_column("final_prediction", post_preds)
    run.log({"eval_table": table})  # 追加の列で最終更新
```

トレーニング ループのように、列は増やさず行だけをバッチで段階的に追加したい場合は、代わりに [`INCREMENTAL` モード]({{< relref path="#INCREMENTAL-mode" lang="ja" >}}) の利用を検討してください。

### INCREMENTAL モード

INCREMENTAL モードでは、機械学習の実験中に、行をバッチでテーブルへログします。長時間のジョブをモニタリングしたい場合や、更新のたびに新しいテーブルをログするのが非効率になるような大きなテーブルを扱う場合に最適です。UI では、ログされるたびに新しい行でテーブルが更新され、run の完了を待たずに最新データを確認できます。インクリメントをステップごとにたどって、時点ごとのテーブルも閲覧できます。

{{% alert %}}
W&B App の Run Workspace には 100 インクリメントの上限があります。100 を超えてログした場合、run の Workspace では直近 100 件のみが表示されます。
{{% /alert %}}

次の例では、`INCREMENTAL` モードでテーブルを作成してログし、その後に新しい行を追加します。テーブルはトレーニング ステップ（`step`）ごとに 1 回ログされる点に注意してください。

{{% alert %}}
以下の例では、データを読み込むプレースホルダー関数 `get_training_batch()`、バッチでモデルを学習するプレースホルダー関数 `train_model_on_batch()`、予測を行うプレースホルダー関数 `predict_on_batch()` を使用しています。ご自身のデータ読み込み・学習・予測関数に置き換えてください。
{{% /alert %}}

```python
import wandb

with wandb.init(project="incremental-table-demo") as run:

    # INCREMENTAL ログ モードのテーブルを作成
    table = wandb.Table(columns=["step", "input", "label", "prediction"],
                        log_mode="INCREMENTAL")

    # トレーニング ループ
    for step in range(get_num_batches()): # プレースホルダー関数
        # バッチ データを読み込む
        inputs, labels = get_training_batch(step) # プレースホルダー関数

        # 学習と予測
        train_model_on_batch(inputs, labels) # プレースホルダー関数
        predictions = predict_on_batch(inputs) # プレースホルダー関数

        # バッチのデータをテーブルに追加
        for input_item, label, prediction in zip(inputs, labels, predictions):
            table.add_data(step, input_item, label, prediction)

        # テーブルをインクリメンタルにログする
        run.log({"training_table": table}, step=step)
```

インクリメンタル ログは、毎回新しいテーブルをログする場合（`log_mode=MUTABLE`）に比べ一般的に計算効率が高いです。ただし多くのインクリメントをログすると、W&B App ではテーブルの全行をレンダリングしない場合があります。run の実行中にテーブル データを更新・閲覧しつつ、後で分析に使える完全なデータも保持したい場合は、テーブルを 2 つ使うことを検討してください。1 つは `INCREMENTAL`、もう 1 つは `IMMUTABLE` のログ モードです。 

次の例は、この目的で `INCREMENTAL` と `IMMUTABLE` のログ モードを組み合わせる方法を示します。

```python
import wandb

with wandb.init(project="combined-logging-example") as run:

    # トレーニング中の効率的な更新のためのインクリメンタル テーブルを作成
    incr_table = wandb.Table(columns=["step", "input", "prediction", "label"],
                            log_mode="INCREMENTAL")

    # トレーニング ループ
    for step in range(get_num_batches()):
        # バッチを処理
        inputs, labels = get_training_batch(step)
        predictions = model.predict(inputs)

        # インクリメンタル テーブルにデータを追加
        for inp, pred, label in zip(inputs, predictions, labels):
            incr_table.add_data(step, inp, pred, label)

        # インクリメンタル更新をログ（最終テーブルと区別するため -incr をサフィックスに付与）
        run.log({"table-incr": incr_table}, step=step)

    # トレーニングの最後に、全データを含む不変テーブルを作成
    # デフォルトの IMMUTABLE モードで完全なデータセットを保持
    final_table = wandb.Table(columns=incr_table.columns, data=incr_table.data, log_mode="IMMUTABLE")
    run.log({"table": final_table})
```

この例では、`incr_table` はトレーニング中に（`log_mode="INCREMENTAL"` で）インクリメンタルにログされます。これにより、新しいデータが処理されるたびにテーブルの更新をログ・閲覧できます。トレーニングの最後に、インクリメンタル テーブルの全データから不変テーブル（`final_table`）を作成します。不変テーブルをログすることで、完全なデータセットを保持し、W&B App で全行を表示できます。 

## 例 

### MUTABLE で評価結果を充実させる

```python
import wandb
import numpy as np

with wandb.init(project="mutable-logging") as run:

    # ステップ 1: 初期予測をログ
    table = wandb.Table(columns=["input", "label", "prediction"], log_mode="MUTABLE")
    inputs, labels = load_eval_data()
    raw_preds = model.predict(inputs)

    for inp, label, pred in zip(inputs, labels, raw_preds):
        table.add_data(inp, label, pred)

    run.log({"eval_table": table})  # 生の予測をログ

    # ステップ 2: 信頼度スコアを追加（例: ソフトマックスの最大値）
    confidences = np.max(raw_preds, axis=1)
    table.add_column("confidence", confidences)
    run.log({"eval_table": table})  # 信頼度情報を追加

    # ステップ 3: 後処理済みの予測を追加
    # （例: 閾値処理や平滑化した出力）
    post_preds = (confidences > 0.7).astype(int)
    table.add_column("final_prediction", post_preds)
    run.log({"eval_table": table})
```

### INCREMENTAL テーブルで run を再開する

run を再開するときも、インクリメンタル テーブルへのログを継続できます:

```python
# run を開始または再開
resumed_run = wandb.init(project="resume-incremental", id="your-run-id", resume="must")

# インクリメンタル テーブルを作成（以前にログしたテーブルのデータを読み戻す必要はありません）
# インクリメントは Table Artifact に引き続き追加されます。
table = wandb.Table(columns=["step", "metric"], log_mode="INCREMENTAL")

# ログを継続
for step in range(resume_step, final_step):
    metric = compute_metric(step)
    table.add_data(step, metric)
    resumed_run.log({"metrics": table}, step=step)

resumed_run.finish()
```

{{% alert %}}
`wandb.Run.define_metric("<table_key>", summary="none")` や `wandb.Run.define_metric("*", summary="none")` を使って、インクリメンタル テーブルで使用しているキーのサマリーをオフにすると、インクリメントは新しいテーブルにログされます。
{{% /alert %}}


### INCREMENTAL でのバッチ トレーニング

```python

with wandb.init(project="batch-training-incremental") as run:

    # インクリメンタル テーブルを作成
    table = wandb.Table(columns=["step", "input", "label", "prediction"], log_mode="INCREMENTAL")

    # 疑似トレーニング ループ
    for step in range(get_num_batches()):
        # バッチ データを読み込む
        inputs, labels = get_training_batch(step)

        # このバッチでモデルを学習
        train_model_on_batch(inputs, labels)

        # モデル推論を実行
        predictions = predict_on_batch(inputs)

        # テーブルにデータを追加
        for input_item, label, prediction in zip(inputs, labels, predictions):
            table.add_data(step, input_item, label, prediction)

        # テーブルの現在の状態をインクリメンタルにログする
        run.log({"training_table": table}, step=step)
```