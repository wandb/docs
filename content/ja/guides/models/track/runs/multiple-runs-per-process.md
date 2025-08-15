---
title: 1 つのプロセス内で複数の run を作成・管理する
description: W&B の reinit 機能を使って、1 つの Python プロセス内で複数の run を管理する
menu:
  default:
    identifier: ja-guides-models-track-runs-multiple-runs-per-process
    parent: what-are-runs
---

1 つの Python プロセス内で複数の run を管理できます。これは、メインとなるプロセスをアクティブに保ちながら、サブタスク用に短命なセカンダリープロセスを作成したいワークフローで役立ちます。主なユースケース例は以下の通りです。

- 1 つのスクリプト内で「メイン」run をアクティブなままにしつつ、評価やサブタスク用に短期間だけ実行する「セカンダリー」run を立ち上げる場合
- 1 ファイル内でサブ実験をまとめてオーケストレーションする場合
- 1 つの「メイン」プロセスから、タスクや期間ごとに異なる複数の run へログを送る場合

通常、W&B では `wandb.init()` を呼び出すたびに、各 Python プロセスにつき 1 つだけアクティブな run が存在すると想定しています。`wandb.init()` を再度呼んだ場合、同じ run を返すか、前の run を終了してから新しい run を開始します（設定によります）。このガイドでは、`reinit` を使い `wandb.init()` の振る舞いを変え、1 つの Python プロセス内で複数の run を有効にする方法を解説します。

{{% alert title="動作要件" %}}
1 つの Python プロセス内で複数の run を管理するには、W&B Python SDK バージョン `v0.19.10` 以上が必要です。
{{% /alert  %}}

## `reinit` のオプション

`reinit` パラメータを使い、`wandb.init()` を複数回呼び出した際の W&B の振る舞いを制御できます。以下の表は有効な引数とその効果です。

| | 説明 | run を新たに作成？ | 主なユースケース例 |
|----------------|----------------|----------------|------------------|
| `create_new` | 現在アクティブな run を終了せずに、`wandb.init()` で新しい run を作成。グローバルな `wandb.Run` は自動的に新しい run へ切り替わりません。各 run オブジェクトを自分で管理する必要があります。詳細は下の [1 プロセス内で複数 run 例]({{< relref path="multiple-runs-per-process/#example-multiple-runs-in-one-process" lang="ja" >}}) を参照してください。 | はい | 並列/同時のプロセス生成・管理に最適です。たとえば、1 つの「メイン」run をアクティブにしつつ「セカンダリー」run の開始・終了を行う場合など。|
| `finish_previous` | 新しい run を `wandb.init()` で作成する前に、すべてのアクティブな run を `run.finish()` で終了する。ノートブック以外の環境でのデフォルトの動作です。 | はい | 順次的なサブプロセスごとに run を分けたい場合に最適 |
| `return_previous` | 最も最近の、未完了の run を返す。ノートブック環境でのデフォルト動作です。 | いいえ | |

{{% alert  %}}
Hugging Face Trainer や Keras コールバック、PyTorch Lightning など、グローバルな run を想定する [W&B インテグレーション]({{< relref path="/guides/integrations/" lang="ja" >}}) では `create_new` モードはサポートされていません。これらのインテグレーションを使う場合、各サブ実験を個別のプロセスで実行してください。
{{% /alert %}}

## `reinit` の指定方法

- `wandb.init()` で `reinit` 引数を直接指定する
   ```python
   import wandb
   wandb.init(reinit="<create_new|finish_previous|return_previous>")
   ```
- `wandb.init()` で、`wandb.Settings` オブジェクトを `settings` パラメータ経由で渡し、Settings 内で `reinit` を指定
   ```python
   import wandb
   wandb.init(settings=wandb.Settings(reinit="<create_new|finish_previous|return_previous>"))
   ```
- `wandb.setup()` でカレントプロセス内のすべての run に対して `reinit` オプションをグローバルに指定  
   1 度だけ振る舞いを設定し、その後の `wandb.init()` すべてに適用したい場合に便利です。
   ```python
   import wandb
   wandb.setup(wandb.Settings(reinit="<create_new|finish_previous|return_previous>"))
   ```
- 環境変数 `WANDB_REINIT` で `reinit` の値を指定する  
   環境変数で指定すると、`wandb.init()` の呼び出しに適用されます。
   ```bash
   export WANDB_REINIT="<create_new|finish_previous|return_previous>"
   ```

以下のコードスニペットは、毎回 `wandb.init()` を呼び出すたびに新しい run を作成するための設定例です:

```python
import wandb

wandb.setup(wandb.Settings(reinit="create_new"))

with wandb.init() as experiment_results_run:
    # この run は各実験の結果を記録するために使います。
    # 親 run として、それぞれの実験の結果をまとめておくイメージです。
    with wandb.init() as run:
        # do_experiment() 関数で、各 run に細かいメトリクスを記録します。
        # また、分けて追跡したい結果用メトリクスを返します。
        experiment_results = do_experiment(run)

        # 各実験の後、その結果を親 run に記録します。
        # 親 run のグラフ上の各点は、各実験の結果に対応しています。
        experiment_results_run.log(experiment_results)
```

## 例: 複数のプロセスを同時に扱う

例えば、スクリプトの期間中ずっと開いたままの「メイン」プロセスを作成し、そこから随時短命な「セカンダリー」プロセスを終了せずに立ち上げたい場合。このパターンは、メイン run でモデルのトレーニングを行い、評価などを別 run で実行したい場合に便利です。

この場合、`reinit="create_new"` を指定して複数の run を初期化します。例えば「Run A」がスクリプト全体でオープンなメインプロセス、「Run B1」「Run B2」などが評価用のセカンダリー run だとします。

高レベルのワークフロー例は以下の通りです:

1. メインプロセス（Run A）を `wandb.init()` で初期化し、トレーニングメトリクスを記録
2. Run B1 を初期化し、データを記録・終了
3. Run A へさらにデータを書き込む
4. Run B2 を初期化し、データ記録・終了
5. Run A へログを継続して記録
6. 最後に Run A を終了

下記の Python コード例でこのワークフローを示します:

```python
import wandb

def train(name: str) -> None:
    """1 回分のトレーニングを独立した W&B run で実行します。

    `reinit="create_new"` 付きの 'with wandb.init()' ブロックを使うことで、
    （メインのトラッキング run など）他の run がすでにアクティブでも、
    このサブ run を作成できます。
    """
    with wandb.init(
        project="my_project",
        name=name,
        reinit="create_new"
    ) as run:
        # 本来ならこのブロックでトレーニング処理を実行します。
        run.log({"train_loss": 0.42})  # 実際のメトリクスに置き換えてください

def evaluate_loss_accuracy() -> (float, float):
    """現在のモデルの損失・Accuracy を返します。
    
    このプレースホルダは実際の評価処理に置き換えてください。
    """
    return 0.27, 0.91  # 例としてのメトリクス値

# 複数のトレーニング・評価工程にわたってアクティブな「メイン」の run を作成
with wandb.init(
    project="my_project",
    name="tracking_run",
    reinit="create_new"
) as tracking_run:
    # 1) サブ run 'training_1' で 1 回トレーニング
    train("training_1")
    loss, accuracy = evaluate_loss_accuracy()
    tracking_run.log({"eval_loss": loss, "eval_accuracy": accuracy})

    # 2) サブ run 'training_2' でもう 1 回トレーニング
    train("training_2")
    loss, accuracy = evaluate_loss_accuracy()
    tracking_run.log({"eval_loss": loss, "eval_accuracy": accuracy})
    
    # 'tracking_run' はこの 'with' ブロック終了時に自動的に終了します。
```

前述の例から重要なポイントは次の 3 つです:

1. `reinit="create_new"` を指定すると、`wandb.init()` を呼ぶたびに新しい run が作成されます。
2. 各 run への参照を自分で保持する必要があります。`wandb.run` は `reinit="create_new"` で新規作成された run を自動的に指しません。`run_a`, `run_b1` などの変数に格納し、必要に応じて `.log()` や `.finish()` を呼んでください。
3. メイン run を開きつつ、サブ run はいつでも好きなタイミングで終了できます。
4. ログが終わった run は `run.finish()` で終了処理を行いましょう。これによりデータのアップロードおよび run の正しいクローズ処理が保証されます。