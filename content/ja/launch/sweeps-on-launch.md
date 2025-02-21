---
title: Create sweeps with W&B Launch
description: Launch でハイパーパラメータ sweep を自動化する方法をご覧ください。
menu:
  launch:
    identifier: ja-launch-sweeps-on-launch
    parent: launch
url: guides/launch/sweeps-on-launch
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1WxLKaJlltThgZyhc7dcZhDQ6cjVQDfil#scrollTo=AFEzIxA6foC7" >}}

W&B Launch でハイパーパラメータ チューニング ジョブ ([Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}})) を作成します。Launch 上の Sweeps では、sweep scheduler が、スイープ対象として指定されたハイパーパラメータとともに Launch Queue にプッシュされます。sweep scheduler は、エージェントによって選択されると起動し、選択されたハイパーパラメータを使用して sweep run を同じキューに起動します。これは、sweep が終了するか停止するまで継続されます。

デフォルトの W&B Sweep スケジューリング エンジンを使用するか、独自のカスタム スケジューラを実装できます。

1. 標準 sweep scheduler: [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) を制御するデフォルトの W&B Sweep スケジューリング エンジンを使用します。使い慣れた `bayes`、`grid`、および `random` の method を利用できます。
2. カスタム sweep scheduler: sweep scheduler をジョブとして実行するように設定します。このオプションを使用すると、完全にカスタマイズできます。標準の sweep scheduler を拡張して、より多くのログを含める方法の例は、以下のセクションにあります。
 
{{% alert %}}
このガイドでは、W&B Launch が以前に設定されていることを前提としています。W&B Launch が設定されていない場合は、Launch ドキュメントの[開始方法]({{< relref path="./#how-to-get-started" lang="ja" >}})セクションを参照してください。
{{% /alert %}}

{{% alert %}}
Launch で Sweeps を初めて使用する場合は、「basic」method を使用して Launch で sweep を作成することをお勧めします。標準の W&B スケジューリング エンジンがニーズを満たさない場合は、Launch スケジューラでカスタム Sweeps を使用してください。
{{% /alert %}}

## W&B 標準スケジューラで sweep を作成する
Launch で W&B Sweeps を作成します。W&B App でインタラクティブに、または W&B CLI でプログラムで sweep を作成できます。スケジューラのカスタマイズ機能など、Launch sweeps の高度な設定を行うには、CLI を使用します。

{{% alert %}}
W&B Launch で sweep を作成する前に、まず sweep するジョブを作成してください。詳細については、[ジョブの作成]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ja" >}}) ページを参照してください。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B app" %}}

W&B App でインタラクティブに sweep を作成します。

1. W&B App で W&B の プロジェクトに移動します。
2. 左側の パネル (ほうきの画像) で Sweeps アイコンを選択します。
3. 次に、**Create Sweep** ボタンを選択します。
4. **Configure Launch 🚀** ボタンをクリックします。
5. **Job** ドロップダウン メニューから、ジョブの名前と、sweep の作成元となるジョブ のバージョンを選択します。
6. **Queue** ドロップダウン メニューを使用して、sweep を実行するキューを選択します。
7. **Job Priority** ドロップダウンを使用して、Launch ジョブ の優先度を指定します。Launch キューが優先順位付けをサポートしていない場合、Launch ジョブ の優先度は「Medium」に設定されます。
8. (オプション) run または sweep scheduler のオーバーライド arg を設定します。たとえば、scheduler のオーバーライドを使用して、scheduler が管理する同時 run の数を `num_workers` を使用して設定します。
9. (オプション) **Destination Project** ドロップダウン メニューを使用して、sweep を保存する プロジェクト を選択します。
10. **Save** をクリックします
11. **Launch Sweep** を選択します。

{{< img src="/images/launch/create_sweep_with_launch.png" alt="" >}}

{{% /tab %}}
{{% tab "CLI" %}}

W&B CLI で Launch を使用して、プログラムで W&B Sweep を作成します。

1. Sweep configuration を作成します。
2. Sweep configuration 内でジョブ の完全な名前を指定します。
3. sweep agent を初期化します。

{{% alert %}}
手順 1 と 3 は、W&B Sweep を作成するときに通常行う手順と同じです。
{{% /alert %}}

たとえば、次の コード スニペット では、ジョブ の値として `'wandb/jobs/Hello World 2:latest'` を指定します。

```yaml
# launch-sweep-config.yaml

job: 'wandb/jobs/Hello World 2:latest'
description: sweep examples using launch jobs

method: bayes
metric:
  goal: minimize
  name: loss_metric
parameters:
  learning_rate:
    max: 0.02
    min: 0
    distribution: uniform
  epochs:
    max: 20
    min: 0
    distribution: int_uniform

# Optional scheduler parameters:

# scheduler:
#   num_workers: 1  # concurrent sweep runs
#   docker_image: <base image for the scheduler>
#   resource: <ie. local-container...>
#   resource_args:  # resource arguments passed to runs
#     env: 
#         - WANDB_API_KEY

# Optional Launch Params
# launch: 
#    registry: <registry for image pulling>
```

sweep configuration の作成方法については、[Define sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration.md" lang="ja" >}}) ページを参照してください。

4. 次に、sweep を初期化します。設定ファイルへのパス、ジョブ キュー の名前、W&B の Entity 、および プロジェクト の名前を指定します。

```bash
wandb launch-sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>
```

W&B Sweeps の詳細については、[Tune Hyperparameters]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) のチャプターを参照してください。

{{% /tab %}}
{{< /tabpane >}}


## カスタム sweep scheduler を作成する
W&B scheduler またはカスタム scheduler を使用して、カスタム sweep scheduler を作成します。

{{% alert %}}
scheduler ジョブ を使用するには、wandb cli のバージョンが >= `0.15.4` である必要があります。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B scheduler" %}}
  W&B sweep スケジューリング ロジックをジョブとして使用して、Launch sweep を作成します。
  
  1. 公開されている wandb/sweep-jobs プロジェクトで Wandb scheduler ジョブ を特定するか、ジョブ 名を使用します:
  `'wandb/sweep-jobs/job-wandb-sweep-scheduler:latest'`
  2. この名前を指す `job` キーを含む追加の `scheduler` ブロックを含む 設定 yaml を構築します。以下の例を参照してください。
  3. 新しい 設定 で `wandb launch-sweep` コマンドを使用します。


設定例:
```yaml
# launch-sweep-config.yaml  
description: Launch sweep config using a scheduler job
scheduler:
  job: wandb/sweep-jobs/job-wandb-sweep-scheduler:latest
  num_workers: 8  # allows 8 concurrent sweep runs

# training/tuning job that the sweep runs will execute
job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
method: grid
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
```
{{% /tab %}}
{{% tab "Custom scheduler" %}}
  カスタム scheduler は、scheduler-job を作成することで作成できます。このガイドの目的のために、ログをより多く提供するために `WandbScheduler` を変更します。

  1. `wandb/launch-jobs` リポジトリ (具体的には: `wandb/launch-jobs/jobs/sweep_schedulers`) をクローンします。
  2. これで、`wandb_scheduler.py` を変更して、目的のログ増加を実現できます。例: 関数 `_poll` にログを追加します。これは、新しい sweep run を起動する前に、ポーリング サイクルごとに 1 回 (設定可能なタイミング) 呼び出されます。
  3. 変更したファイルを実行してジョブを作成します。`python wandb_scheduler.py --project <project> --entity <entity> --name CustomWandbScheduler`
  4. 作成されたジョブ の名前を UI または前の呼び出しの出力で確認します。これは、(特に指定がない限り) コード アーティファクト ジョブ になります。
  5. 次に、scheduler が新しいジョブ を指す sweep configuration を作成します。

```yaml
...
scheduler:
  job: '<entity>/<project>/job-CustomWandbScheduler:latest'
...
```

{{% /tab %}}
{{% tab "Optuna scheduler" %}}

  Optuna は、(W&B と同様に) 特定のモデルに最適なハイパーパラメータを見つけるために、さまざまな アルゴリズム を使用するハイパーパラメータ最適化 フレームワーク です。[サンプリング アルゴリズム](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html)に加えて、Optuna は、パフォーマンスの低い run を早期に終了するために使用できるさまざまな[枝刈り アルゴリズム](https://optuna.readthedocs.io/en/stable/reference/pruners.html)も提供します。これは、多数の run を実行する場合に特に役立ち、時間とリソースを節約できます。クラスは高度に設定可能で、設定ファイルの `scheduler.settings.pruner/sampler.args` ブロックに必要な パラメータ を渡すだけです。

Optuna のスケジューリング ロジックをジョブ とともに使用して、Launch sweep を作成します。

1. まず、独自のジョブ を作成するか、事前構築済みの Optuna スケジューラ イメージ ジョブ を使用します。
    * 独自のジョブ を作成する方法の例については、[`wandb/launch-jobs`](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers) リポジトリを参照してください。
    * 事前構築済みの Optuna イメージを使用するには、`wandb/sweep-jobs` プロジェクトの `job-optuna-sweep-scheduler` に移動するか、ジョブ 名 `wandb/sweep-jobs/job-optuna-sweep-scheduler:latest` を使用できます。

2. ジョブ を作成したら、sweep を作成できます。Optuna scheduler ジョブ を指す `job` キーを含む `scheduler` ブロックを含む sweep 設定を構築します (以下の例)。

```yaml
  # optuna_config_basic.yaml
  description: A basic Optuna scheduler
  job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
  run_cap: 5
  metric:
    name: epoch/val_loss
    goal: minimize

  scheduler:
    job: wandb/sweep-jobs/job-optuna-sweep-scheduler:latest
    resource: local-container  # required for scheduler jobs sourced from images
    num_workers: 2

    # optuna specific settings
    settings:
      pruner:
        type: PercentilePruner
        args:
          percentile: 25.0  # kill 75% of runs
          n_warmup_steps: 10  # pruning turned off for first x steps

  parameters:
    learning_rate:
      min: 0.0001
      max: 0.1
  ```


  3. 最後に、launch-sweep コマンドを使用して、アクティブなキューに sweep を Launch します。
  
  ```bash
  wandb launch-sweep <config.yaml> -q <queue> -p <project> -e <entity>
  ```

  Optuna sweep scheduler ジョブ の正確な実装については、[wandb/launch-jobs](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers/optuna_scheduler/optuna_scheduler.py) を参照してください。Optuna scheduler で可能なことのその他の例については、[wandb/examples](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/optuna-scheduler) を確認してください。
{{% /tab %}}
{{< /tabpane >}}

 カスタム sweep scheduler ジョブ で可能なことの例は、`jobs/sweep_schedulers` の下の [wandb/launch-jobs](https://github.com/wandb/launch-jobs) リポジトリで入手できます。このガイドでは、一般公開されている **Wandb Scheduler Job** の使用方法と、カスタム sweep scheduler ジョブ を作成するための プロセス について説明します。

 ## Launch で Sweeps を再開する方法
  以前に Launch された sweep から Launch-sweep を再開することも可能です。ハイパー パラメータ と トレーニング ジョブ は変更できませんが、スケジューラ固有の パラメータ と、プッシュ先のキュー は変更できます。

{{% alert %}}
最初の sweep で 'latest' などの エイリアス を持つ トレーニング ジョブ を使用した場合、最後の run 以降に最新のジョブ バージョンが変更されていると、再開すると異なる 結果 になる可能性があります。
{{% /alert %}}

  1. 以前に実行された Launch sweep の sweep 名/ID を特定します。sweep ID は 8 文字の文字列 (たとえば、`hhd16935`) で、W&B App の プロジェクト で確認できます。
  2. scheduler パラメータ を変更する場合は、更新された 設定 ファイルを構築します。
  3. ターミナル で、次のコマンドを実行します。`<` および `>` で囲まれたコンテンツを自分の情報に置き換えます。

```bash
wandb launch-sweep <optional config.yaml> --resume_id <sweep id> --queue <queue_name>
```
