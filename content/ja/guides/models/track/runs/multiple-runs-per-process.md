---
title: 1 つのプロセス内で複数の run を作成・管理する
description: W&B の reinit 機能を使って、1つの Python プロセス内で複数の Run を管理する
menu:
  default:
    identifier: multiple-runs
    parent: what-are-runs
---

単一の Python プロセス内で複数の run を管理できます。これは、メインのプロセスをアクティブに保ちながら、短命なサブタスク用のセカンダリプロセスを作成したいワークフローで役立ちます。主なユースケース例は以下の通りです。

- スクリプト全体で 1 つの「プライマリ」run をアクティブに保ちつつ、評価やサブタスクごとに短命な「セカンダリ」run を立ち上げる場合  
- 1 つのファイルでサブ実験をまとめて管理する場合  
- 1 つの「メイン」プロセスから複数の run にログを送信し、それぞれ異なるタスクや期間を表現する場合

デフォルトでは、W&B は各 Python プロセスで `wandb.init()` を呼ぶと 1 つだけアクティブな run があるとみなします。`wandb.init()` を再び呼び出すと、同じ run を返すか、設定によっては前の run を終了させて新しい run を開始します。このガイドでは、`reinit` を使って `wandb.init()` の振る舞いを変更し、1 つの Python プロセス内で複数の run を有効化する方法を紹介します。

{{% alert title="Requirements" %}}
単一の Python プロセス内で複数 run を管理するには、W&B Python SDK バージョン `v0.19.10` 以上が必要です。
{{% /alert  %}}

## `reinit` オプション

W&B が `wandb.init()` を複数回呼んだ際にどのように扱うかは、`reinit` パラメータで設定できます。次の表は有効な引数とその効果の概要です。

| | 説明 | run を作成するか | 主なユースケース |
|----------------|----------------|----------------| -----------------|
| `create_new` | 既存でアクティブな run を終了せずに `wandb.init()` で新しい run を作成します。W&B はグローバルな `wandb.Run` を自動で新規 run に切り替えません。各 run オブジェクトを自分で保持してください。詳しくは[1プロセスで複数 run の例]({{< relref "multiple-runs-per-process/#example-multiple-runs-in-one-process" >}})を参照してください。 | 作成する | メイン run を起動したままセカンダリ run を同時並行で管理したい場合などに最適です。|
| `finish_previous` | 新規 run を `wandb.init()` で作る前に、`run.finish()` を使ってすべてのアクティブな run を終了します（ノートブック以外の環境でのデフォルト挙動）。 | 作成する | サブプロセスを個別の run として順番に区切りたい場合に便利です。|
| `return_previous` | 一番新しい未完了の run を返します（ノートブック環境でのデフォルト挙動）。| 作成しない | |

{{% alert  %}}
W&B の [W&B Integrations]({{< relref "/guides/integrations/" >}}) のうち、グローバル run が1つであることを前提とする（たとえば Hugging Face Trainer・Keras のコールバック・PyTorch Lightning など）場合は `create_new` モードはサポートされません。これらを使うときは各サブ実験を別プロセスで実行してください。
{{% /alert %}}

## `reinit` の指定方法

- `wandb.init()` の引数として直接 `reinit` を指定:

   ```python
   import wandb
   wandb.init(reinit="<create_new|finish_previous|return_previous>")
   ```

- `wandb.init()` の `settings` パラメータに `wandb.Settings` オブジェクトを渡し、その中で `reinit` を指定:

   ```python
   import wandb
   wandb.init(settings=wandb.Settings(reinit="<create_new|finish_previous|return_previous>"))
   ```

- プロセス全体の挙動をまとめて設定したい場合は `wandb.setup()` でグローバル指定:

   ```python
   import wandb
   wandb.setup(wandb.Settings(reinit="<create_new|finish_previous|return_previous>"))
   ```

- 環境変数 `WANDB_REINIT` で `reinit` の値を指定。環境変数による設定は、そのプロセス内のすべての `wandb.init()` に適用されます。

   ```bash
   export WANDB_REINIT="<create_new|finish_previous|return_previous>"
   ```

以下のコードスニペットは、`wandb.init()` を呼ぶたびに毎回新しい run を作成する W&B のセットアップ例の概要です。

```python
import wandb

wandb.setup(wandb.Settings(reinit="create_new"))

with wandb.init() as experiment_results_run:
    # この run は各実験の結果をまとめてログします
    # 親 run（各実験の成果を集約する run）として利用できます
      with wandb.init() as run:
         # do_experiment() 関数で詳細なメトリクスを
         # 指定 run に記録し、個別追跡したい
         # 結果メトリクスを返します
         experiment_results = do_experiment(run)

         # 各実験後、その結果を親 run に記録
         # 親 run のチャートの各ポイントは
         # 1つの実験結果に対応します
         experiment_results_run.log(experiment_results)
```

## 例：並列プロセス

例えば、スクリプトの寿命中ずっと開いているプライマリなプロセスを作成し、適宜セカンダリの短命プロセスを生成（ただしプライマリプロセスは終了させない）といったことをしたい場合があります。このパターンは、プライマリ run でモデルをトレーニングしつつ、別の run で評価や追加作業を別個に記録したい場面で役立ちます。

こうした場合は `reinit="create_new"` を指定して複数の run を初期化します。たとえば、「Run A」をスクリプト全体で維持し、評価などのタスクは「Run B1」「Run B2」などの短命セカンダリ run で行う、というイメージです。

ワークフローの概要例：

1. プライマリプロセス（Run A）を `wandb.init()` で初期化し、トレーニングのメトリクスを記録  
2. Run B1 を初期化（`wandb.init()`）、データをログし終了  
3. 再度 Run A へデータを記録  
4. Run B2 を初期化・データ記録・終了  
5. また Run A に記録  
6. 最後に Run A を終了

このワークフローを示す Python コード例は下記の通りです。

```python
import wandb

def train(name: str) -> None:
    """1回分のトレーニングを独立した W&B run として実行します

    'with wandb.init()' ブロックと `reinit="create_new"` を使えば、
    （プライマリの tracking run がアクティブな場合でも）
    このサブ run を新規に作成できます
    """
    with wandb.init(
        project="my_project",
        name=name,
        reinit="create_new"
    ) as run:
        # 実際のスクリプトではこのブロック内でトレーニング処理を実行します
        run.log({"train_loss": 0.42})  # 実際のメトリクスに置き換えてください

def evaluate_loss_accuracy() -> (float, float):
    """現在のモデルの損失値と精度を返します
    
    実際の評価処理に置き換えてください
    """
    return 0.27, 0.91  # 例としてのメトリクス値

# 複数回の train/eval の期間中ずっとアクティブな「プライマリ」run を作成します
with wandb.init(
    project="my_project",
    name="tracking_run",
    reinit="create_new"
) as tracking_run:
    # 1) 'training_1' というサブ run でまず1回トレーニング
    train("training_1")
    loss, accuracy = evaluate_loss_accuracy()
    tracking_run.log({"eval_loss": loss, "eval_accuracy": accuracy})

    # 2) 再び 'training_2' というサブ run でトレーニング
    train("training_2")
    loss, accuracy = evaluate_loss_accuracy()
    tracking_run.log({"eval_loss": loss, "eval_accuracy": accuracy})
    
    # 'tracking_run' はこの 'with' ブロック終了時点で自動的に完了します
```

上記例から読み取れる主なポイントは次の4つです：

1. `reinit="create_new"` を使うと `wandb.init()` のたびに新しい run が作成されます
2. 各 run の参照を自分で管理します。`wandb.run` は `reinit="create_new"` で作られた新 run を自動で指しません。変数 `run_a`, `run_b1` などに格納し、必要に応じて `.log()` や `.finish()` を実行してください
3. サブ run を好きなタイミングで終了し、プライマリ run は好きなだけオープンのままにできます
4. ログ取りが終わった run は `run.finish()` で明示的に完了させましょう。データアップロードと run 終了処理が正しく行われます