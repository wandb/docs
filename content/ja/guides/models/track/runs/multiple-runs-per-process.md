---
title: 単一のプロセス内で複数の runs を作成・管理する
description: W&B の reinit 機能を使い、単一の Python プロセス内で複数の runs を管理する
menu:
  default:
    identifier: ja-guides-models-track-runs-multiple-runs-per-process
    parent: what-are-runs
---

単一の Python プロセス内で複数の run を管理します。これは、メインのプロセスを生かしたまま、サブタスク用に短命のセカンダリプロセスを作るようなワークフローで役立ちます。ユースケースの例:

- スクリプト全体で 1 つの「primary」run を生かしつつ、評価やサブタスク用に短命の「secondary」run を立ち上げる。  
- 1 ファイル内でサブ実験をオーケストレーションする。  
- 1 つの「main」プロセスから、タスクや期間の異なる複数の run にログする。

デフォルトでは、`wandb.init()` を呼ぶと W&B は各 Python プロセスに同時に 1 つだけアクティブな run があると想定します。`wandb.init()` を再度呼ぶと、設定に応じて同じ run を返すか、古い run を終了してから新しい run を開始します。本ガイドでは、`reinit` を使って `wandb.init()` の振る舞いを変更し、1 つの Python プロセス内で複数の run を有効化する方法を説明します。

{{% alert title="要件" %}}
単一の Python プロセス内で複数の run を管理するには、W&B Python SDK の バージョン `v0.19.10` 以降が必要です。
{{% /alert  %}}

## `reinit` のオプション

`reinit` パラメータを使って、W&B が `wandb.init()` を複数回呼び出したときの扱いを設定します。以下は有効な引数とその効果の一覧です:

| | 説明 | run を作成するか | ユースケースの例 |
|----------------|----------------|----------------| -----------------|
| `create_new` | 既存のアクティブな run を終了せずに、`wandb.init()` で新しい run を作成します。W&B はグローバルな `wandb.Run` を新しい run に自動で切り替えません。各 run オブジェクトは自分で保持する必要があります。詳細は下記の [1 つのプロセスで複数の run の例]({{< relref path="multiple-runs-per-process/#example-multiple-runs-in-one-process" lang="ja" >}}) を参照してください。  | Yes | 複数のプロセスを同時に作成・管理したい場合に最適。例えば、アクティブなままの「primary」run と、必要に応じて開始・終了する「secondary」run。|
| `finish_previous` | 新しい run を `wandb.init()` で作成する前に、アクティブな run をすべて `run.finish()` で終了します。ノートブック 以外の 環境 のデフォルトの振る舞いです。 | Yes | 逐次的なサブプロセスを個別の run に分けたい場合に最適。 |
| `return_previous` | 直近の未終了の run を返します。ノートブック 環境 のデフォルトの振る舞いです。 | No | |

{{% alert  %}}
単一のグローバル run を前提とする [W&B Integrations]({{< relref path="/guides/integrations/" lang="ja" >}})（Hugging Face Trainer、Keras callbacks、PyTorch Lightning など）では、`create_new` モードはサポートされません。これらのインテグレーションを使う場合は、各サブ実験を別プロセスで実行してください。
{{% /alert %}}

## `reinit` の指定方法

- `reinit` 引数を直接指定して `wandb.init()` を使う:
   ```python
   import wandb
   wandb.init(reinit="<create_new|finish_previous|return_previous>")
   ```
- `wandb.init()` の `settings` パラメータに `wandb.Settings` オブジェクトを渡し、その中で `reinit` を指定する:

   ```python
   import wandb
   wandb.init(settings=wandb.Settings(reinit="<create_new|finish_previous|return_previous>"))
   ```

- 現在のプロセス内のすべての run に対して `reinit` をグローバルに設定するには `wandb.setup()` を使う。この方法は、一度だけ振る舞いを設定し、その後のすべての `wandb.init()` 呼び出しに適用したいときに便利です。

   ```python
   import wandb
   wandb.setup(wandb.Settings(reinit="<create_new|finish_previous|return_previous>"))
   ```

- 環境変数 `WANDB_REINIT` に `reinit` の希望値を指定する。環境変数を定義すると、そのプロセス内の `wandb.init()` 呼び出しに適用されます。

   ```bash
   export WANDB_REINIT="<create_new|finish_previous|return_previous>"
   ```

次のコードスニペットは、`wandb.init()` を呼ぶたびに新しい run を作成するように W&B をセットアップする全体像を示します:

```python
import wandb

wandb.setup(wandb.Settings(reinit="create_new"))

with wandb.init() as experiment_results_run:
    # この run は各実験の結果をログするために使われます。
    # 親 run として、結果を集約するイメージです
      with wandb.init() as run:
         # do_experiment() は与えられた run に詳細なメトリクスを
         # ログし、別途トラッキングしたい結果メトリクスを返します。
         experiment_results = do_experiment(run)

         # 各実験のあとに、その結果を親 run にログします。
         # 親 run のチャートの各ポイントは、1 回の実験の結果に対応します。
         experiment_results_run.log(experiment_results)
```

## 例: 並行プロセス

スクリプトのライフサイクル全体で開いたままにする primary プロセスを作り、定期的に短命の secondary プロセスを起動しつつ、primary プロセスは終了しないようにしたいとします。例えば、primary の run で モデル を学習しつつ、評価やその他の処理は別の run で行いたい場合に有用です。

これを実現するには、`reinit="create_new"` を使って複数の run を初期化します。この例では、スクリプト全体で開いたままにする primary プロセスを「Run A」、評価などのタスク用に短命で起動する secondary run を「Run B1」「Run B2」とします。 

ハイレベルなワークフローは次のようになります:

1. `wandb.init()` で primary プロセスの Run A を初期化し、トレーニングのメトリクスをログする。  
2. Run B1 を初期化（`wandb.init()`）し、データをログしてから終了する。  
3. Run A にさらにデータをログする。  
4. Run B2 を初期化し、データをログしてから終了する。  
5. Run A へログを続ける。  
6. 最後に、終わりで Run A を終了する。

以下の Python コード例はこのワークフローを示します:

```python
import wandb

def train(name: str) -> None:
    """Perform one training iteration in its own W&B run.

    Using a 'with wandb.init()' block with `reinit="create_new"` ensures that
    this training sub-run can be created even if another run (like our primary
    tracking run) is already active.
    """
    with wandb.init(
        project="my_project",
        name=name,
        reinit="create_new"
    ) as run:
        # 実際のスクリプトでは、このブロック内で学習ステップを実行します。
        run.log({"train_loss": 0.42})  # 実際のメトリクスに置き換えてください

def evaluate_loss_accuracy() -> (float, float):
    """Returns the current model's loss and accuracy.
    
    Replace this placeholder with your real evaluation logic.
    """
    return 0.27, 0.91  # メトリクス値の例

# 複数回の学習/評価ステップの間、アクティブなままにする「primary」run を作成します。
with wandb.init(
    project="my_project",
    name="tracking_run",
    reinit="create_new"
) as tracking_run:
    # 1) 'training_1' というサブ-run で 1 回学習
    train("training_1")
    loss, accuracy = evaluate_loss_accuracy()
    tracking_run.log({"eval_loss": loss, "eval_accuracy": accuracy})

    # 2) 'training_2' というサブ-run で再度学習
    train("training_2")
    loss, accuracy = evaluate_loss_accuracy()
    tracking_run.log({"eval_loss": loss, "eval_accuracy": accuracy})
    
    # この 'with' ブロックを抜けると 'tracking_run' は自動で終了します。
```

前の例からの重要なポイントは次の 3 つです:

1. `reinit="create_new"` は、`wandb.init()` を呼ぶたびに新しい run を作成します。
2. 各 run への参照を保持します。`wandb.run` は `reinit="create_new"` で作成された新しい run を自動で指しません。`run_a`、`run_b1` などの変数に新しい run を保存し、必要に応じてそれらのオブジェクトに対して `.log()` や `.finish()` を呼びます。
3. primary の run を開いたまま、サブ-run はいつでも任意のタイミングで終了できます。
4. ログを終えた run は `run.finish()` で終了しましょう。これにより、すべてのデータがアップロードされ、run が正しくクローズされます。