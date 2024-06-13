---
description: ハイパーパラメータスイープをLaunchで自動化する方法を発見する。
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Sweeps on Launch

<CTAButtons colabLink="https://colab.research.google.com/drive/1WxLKaJlltThgZyhc7dcZhDQ6cjVQDfil#scrollTo=AFEzIxA6foC7"/>

W&B Launchでハイパーパラメータチューニングジョブ（[sweeps](../sweeps/intro.md)）を作成します。Launchでのsweepsでは、指定されたハイパーパラメータを持つsweepスケジューラがLaunch Queueにプッシュされます。エージェントがスケジューラをピックアップすると、選択されたハイパーパラメータでsweep runsが同じキューに投入されます。これはsweepが終了するか停止するまで続きます。

デフォルトのW&B Sweepスケジューリングエンジンを使用するか、カスタムスケジューラを実装することができます：

1. 標準のsweepスケジューラ: デフォルトのW&B Sweepスケジューリングエンジンを使用して[W&B Sweeps](../sweeps/intro.md)を制御します。おなじみの`bayes`、`grid`、`random`メソッドが利用可能です。
2. カスタムsweepスケジューラ: ジョブとしてsweepスケジューラを設定します。このオプションでは完全なカスタマイズが可能です。標準のsweepスケジューラを拡張してより多くのログを含める方法の例は、以下のセクションにあります。

:::note
このガイドは、W&B Launchが事前に設定されていることを前提としています。W&B Launchが設定されていない場合は、Launchドキュメントの[開始方法](./intro.md#how-to-get-started)セクションを参照してください。
:::

:::tip
初めてLaunchでsweepsを使用する場合は、「basic」メソッドを使用してsweepを作成することをお勧めします。標準のW&Bスケジューリングエンジンがニーズに合わない場合は、カスタムsweeps on launchスケジューラを使用してください。
:::

## W&B標準スケジューラでsweepを作成する
LaunchでW&B Sweepsを作成します。W&B Appを使用してインタラクティブにsweepを作成するか、W&B CLIを使用してプログラム的に作成できます。スケジューラのカスタマイズを含むLaunch sweepsの高度な設定には、CLIを使用してください。

:::info
W&B Launchでsweepを作成する前に、まずsweep対象のジョブを作成してください。詳細は[Create a Job](./create-launch-job.md)ページを参照してください。
:::

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="app">
W&B Appを使用してインタラクティブにsweepを作成します。

1. W&B Appで自分のW&Bプロジェクトに移動します。
2. 左側のパネルでsweepsアイコン（ほうきの画像）を選択します。
3. 次に、**Create Sweep**ボタンを選択します。
4. **Configure Launch 🚀**ボタンをクリックします。
5. **Job**ドロップダウンメニューから、sweepを作成したいジョブの名前とバージョンを選択します。
6. **Queue**ドロップダウンメニューを使用して、sweepを実行するキューを選択します。
8. **Job Priority**ドロップダウンを使用して、launchジョブの優先順位を指定します。launchキューが優先順位をサポートしていない場合、launchジョブの優先順位は「Medium」に設定されます。
8. （オプション）runまたはsweepスケジューラのオーバーライド引数を設定します。例えば、スケジューラのオーバーライドを使用して、スケジューラが管理する同時実行runの数を`num_workers`で設定します。
9. （オプション）**Destination Project**ドロップダウンメニューを使用して、sweepを保存するプロジェクトを選択します。
10. **Save**をクリックします。
11. **Launch Sweep**を選択します。

![](/images/launch/create_sweep_with_launch.png)

  </TabItem>
  <TabItem value="cli">

W&B CLIを使用してプログラム的にLaunchでW&B Sweepを作成します。

1. Sweep構成を作成します。
2. Sweep構成内に完全なジョブ名を指定します。
3. Sweepエージェントを初期化します。

:::info
ステップ1と3は、通常のW&B Sweepを作成する際の手順と同じです。
:::

例えば、以下のコードスニペットでは、ジョブの値として`'wandb/jobs/Hello World 2:latest'`を指定しています：

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

Sweep構成の作成方法については、[Define sweep configuration](../sweeps/define-sweep-configuration.md)ページを参照してください。

4. 次に、sweepを初期化します。設定ファイルのパス、ジョブキューの名前、W&Bエンティティ、およびプロジェクト名を指定します。

```bash
wandb launch-sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>
```

W&B Sweepsの詳細については、[Tune Hyperparameters](../sweeps/intro.md)章を参照してください。

</TabItem>

</Tabs>

## カスタムsweepスケジューラを作成する
W&Bスケジューラまたはカスタムスケジューラを使用してカスタムsweepスケジューラを作成します。

:::info
スケジューラジョブの使用には、wandb cliバージョン >= `0.15.4`が必要です。
:::

<Tabs
  defaultValue="wandb-scheduler"
  values={[
    {label: 'Wandb scheduler', value: 'wandb-scheduler'},
    {label: 'Optuna scheduler', value: 'optuna-scheduler'},
    {label: 'Custom scheduler', value: 'custom-scheduler'},
  ]}>
    <TabItem value="wandb-scheduler">

  W&B sweepスケジューリングロジックをジョブとして使用してlaunch sweepを作成します。
  
  1. 公開されているwandb/sweep-jobsプロジェクトでWandbスケジューラジョブを特定するか、ジョブ名を使用します：
  `'wandb/sweep-jobs/job-wandb-sweep-scheduler:latest'`
  2. この名前を指す`job`キーを含む追加の`scheduler`ブロックを持つ構成yamlを作成します。以下の例を参照してください。
  3. 新しい構成で`wandb launch-sweep`コマンドを使用します。

例の構成：
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

  </TabItem>
  <TabItem value="custom-scheduler">

  カスタムスケジューラは、スケジューラジョブを作成することで作成できます。このガイドの目的のために、`WandbScheduler`を変更してより多くのログを提供します。

  1. `wandb/launch-jobs`リポジトリをクローンします（具体的には：`wandb/launch-jobs/jobs/sweep_schedulers`）。
  2. 望ましいログの増加を達成するために`wandb_scheduler.py`を変更します。例：関数`_poll`にログを追加します。これは各ポーリングサイクル（設定可能なタイミング）ごとに一度呼び出され、新しいsweep runsを開始する前に実行されます。
  3. 修正されたファイルを実行してジョブを作成します：`python wandb_scheduler.py --project <project> --entity <entity> --name CustomWandbScheduler`
  4. 作成されたジョブの名前を特定します。これはUIまたは前の呼び出しの出力で確認できます。特に指定がない限り、これはコードアーティファクトジョブになります。
  5. スケジューラが新しいジョブを指すようにsweep構成を作成します！

```yaml
...
scheduler:
  job: '<entity>/<project>/job-CustomWandbScheduler:latest'
...
```

  </TabItem>
  <TabItem value="optuna-scheduler">

  Optunaは、与えられたモデルに対して最適なハイパーパラメータを見つけるためにさまざまなアルゴリズムを使用するハイパーパラメータ最適化フレームワークです（W&Bと類似しています）。[サンプリングアルゴリズム](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html)に加えて、Optunaは[プルーニングアルゴリズム](https://optuna.readthedocs.io/en/stable/reference/pruners.html)も提供しており、パフォーマンスの低いrunを早期に終了させることができます。これは、多数のrunを実行する場合に特に有用で、時間とリソースを節約できます。クラスは高度に設定可能で、期待されるパラメータを`scheduler.settings.pruner/sampler.args`ブロックに渡すだけです。

Optunaのスケジューリングロジックを使用してlaunch sweepを作成します。

1. まず、自分のジョブを作成するか、事前に構築されたOptunaスケジューライメージジョブを使用します。
    * 自分のジョブを作成する方法の例については、[`wandb/launch-jobs`](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers)リポジトリを参照してください。
    * 事前に構築されたOptunaイメージを使用するには、`wandb/sweep-jobs`プロジェクトの`job-optuna-sweep-scheduler`に移動するか、ジョブ名を使用します：`wandb/sweep-jobs/job-optuna-sweep-scheduler:latest`。

2. ジョブを作成した後、sweepを作成します。Optunaスケジューラジョブを指す`job`キーを含む`scheduler`ブロックを含むsweep構成を作成します（以下の例を参照）。

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
          n_warmup_steps: 10  # pruning disabled for first x steps

  parameters:
    learning_rate:
      min: 0.0001
      max: 0.1
  ```

3. 最後に、launch-sweepコマンドを使用してアクティブなキューにsweepを投入します：
  
```bash
wandb launch-sweep <config.yaml> -q <queue> -p <project> -e <entity>
```

Optuna sweepスケジューラジョブの正確な実装については、[wandb/launch-jobs](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers/optuna_scheduler/optuna_scheduler.py)を参照してください。Optunaスケジューラで可能なことの詳細については、[wandb/examples](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/optuna-scheduler)をチェックしてください。

</TabItem>
</Tabs>

カスタムsweepスケジューラジョブで可能なことの例は、`jobs/sweep_schedulers`の下にある[wandb/launch-jobs](https://github.com/wandb/launch-jobs)リポジトリで利用できます。このガイドでは、公開されている**Wandb Scheduler Job**の使用方法と、カスタムsweepスケジューラジョブを作成するプロセスを示しています。

## Launchでsweepsを再開する方法
以前に開始されたsweepからlaunch-sweepを再開することも可能です。ハイパーパラメータやトレーニングジョブは変更できませんが、スケジューラ固有のパラメータやプッシュ先のキューは変更できます。

:::info
最初のsweepが「latest」のようなエイリアスを持つトレーニングジョブを使用していた場合、再開すると、最新のジョブバージョンが前回のrun以降に変更されている場合、異なる結果になる可能性があります。
:::

1. 以前に実行されたlaunch sweepのsweep名/IDを特定します。sweep IDは8文字の文字列（例：`hhd16935`）で、W&B Appのプロジェクトで確認できます。
2. スケジューラパラメータを変更する場合は、更新された構成ファイルを作成します。
3. 端末で以下のコマンドを実行します。"<"と">"で囲まれた内容を自分の情報に置き換えます：

```bash
wandb launch-sweep <optional config.yaml> --resume_id <sweep id> --queue <queue_name>
```