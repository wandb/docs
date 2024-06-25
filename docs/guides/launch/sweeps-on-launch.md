---
description: W&B Launchでハイパーパラメータのsweepsを自動化する方法を発見する。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# Sweeps on Launch

<CTAButtons colabLink="https://colab.research.google.com/drive/1WxLKaJlltThgZyhc7dcZhDQ6cjVQDfil#scrollTo=AFEzIxA6foC7"/>

W&B Launchでハイパーパラメータチューニングジョブ ([sweeps](../sweeps/intro.md)) を作成します。Launchでsweepsを使用すると、指定されたハイパーパラメータを使用してsweepスケジューラがLaunch Queueにプッシュされます。エージェントがこれをピックアップすると、選択されたハイパーパラメータでsweep runsを同じキューに起動します。これがsweepが終了するか停止するまで続きます。

デフォルトのW&B Sweepスケジューリングエンジンを使用するか、独自のカスタムスケジューラを実装することができます：

1. 標準のsweepスケジューラ：W&B Sweepsを制御するデフォルトのW&B Sweepスケジューリングエンジンを使用します。`bayes`、`grid`、`random`メソッドが利用可能です。
2. カスタムsweepスケジューラ：ジョブとして稼働するようにsweepスケジューラを設定します。このオプションにより、完全なカスタマイズが可能になります。標準のsweepスケジューラを拡張してログを追加する方法の例は、以下のセクションにあります。

:::note
このガイドは、W&B Launchが事前に設定されていることを前提としています。もしW&B Launchが設定されていない場合は、Launchドキュメントの[開始方法](./intro.md#how-to-get-started)セクションを参照してください。
:::

:::tip
初めてLaunchのsweepsを使用する場合は、「basic」メソッドを使用してsweepを作成することをおすすめします。標準のW&Bスケジューリングエンジンがニーズを満たさない場合は、カスタムのLaunch sweepsスケジューラを使用してください。
:::

## W&B標準スケジューラを使用してsweepを作成する
Launchを使用してW&B Sweepsを作成します。W&B Appを使用してインタラクティブにsweepを作成するか、W&B CLIを使用してプログラム的に作成します。Launch sweepsの高度な設定（スケジューラのカスタマイズを含む）には、CLIを使用してください。

:::info
W&B Launchを使用してsweepを作成する前に、まずsweepを実行するジョブを作成してください。詳細については、[ジョブの作成](./create-launch-job.md)ページを参照してください。
:::


<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="app">
W&B Appを使用してインタラクティブにsweepを作成する。

1. W&B Appで自分のW&Bプロジェクトに移動します。
2. 左パネルのsweepsアイコン（ほうきの画像）を選択します。
3. 次に、**Create Sweep**ボタンを選択します。
4. **Configure Launch 🚀**ボタンをクリックします。
5. **Job**ドロップダウンメニューから、sweepを作成するジョブの名前とジョブバージョンを選択します。
6. **Queue**ドロップダウンメニューを使用して、スウィープを実行するキューを選択します。
8. **Job Priority**ドロップダウンを使用して、Launchジョブの優先順位を指定します。Launchキューが優先順位をサポートしていない場合、Launchジョブの優先順位は「Medium」に設定されます。
8. （オプション）runまたはsweepスケジューラのオーバーライドの引数を構成します。例えば、スケジューラのオーバーライドを使用して、`num_workers`を使用してスケジューラが管理する同時実行数を設定します。
9. （オプション）**Destination Project**ドロップダウンメニューを使用してsweepを保存するプロジェクトを選択します。
10. **Save**をクリックします。
11. **Launch Sweep**を選択します。

![](/images/launch/create_sweep_with_launch.png)

  </TabItem>
  <TabItem value="cli">

W&B CLIを使用してプログラム的にW&B SweepをLaunchで作成する。

1. Sweep設定を作成する
2. sweep設定内で完全なジョブ名を指定する
3. sweep agentを初期化する

:::info
ステップ1と3は、通常、W&B Sweepを作成する際に行うのと同じ手順です。
:::

例えば、以下のコードスニペットでは、ジョブ値として`'wandb/jobs/Hello World 2:latest'`を指定しています：

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

sweep設定方法については、[Define sweep configuration](../sweeps/define-sweep-configuration.md)ページを参照してください。

4. 次に、sweepを初期化します。設定ファイルのパス、ジョブキューの名前、自分のW&B entity、およびプロジェクトの名前を指定します。

```bash
wandb launch-sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>
```

W&B Sweepsの詳細については、[Tune Hyperparameters](../sweeps/intro.md)チャプターを参照してください。


</TabItem>

</Tabs>


## カスタムsweepスケジューラを作成する
W&Bスケジューラまたはカスタムスケジューラを使用してカスタムスイープスケジューラを作成します。

:::info
スケジューラジョブを使用するには、wandb CLI バージョン >= `0.15.4`が必要です。
:::

<Tabs
  defaultValue="wandb-scheduler"
  values={[
    {label: 'Wandb scheduler', value: 'wandb-scheduler'},
    {label: 'Optuna scheduler', value: 'optuna-scheduler'},
    {label: 'Custom scheduler', value: 'custom-scheduler'},
  ]}>
    <TabItem value="wandb-scheduler">

W&B sweepスケジューリングロジックをジョブとして使用してLaunch sweepを作成します。

1. 公開されているwandb/sweep-jobsプロジェクトのWandb schedulerジョブを特定するか、ジョブ名を使用します：`'wandb/sweep-jobs/job-wandb-sweep-scheduler:latest'`
2. `scheduler`ブロックとこの名前を指す`job`キーを含む設定yamlを作成します（下の例を参照）。
3. 新しい設定を使って`wandb launch-sweep`コマンドを使用します。

設定例：
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

カスタムスケジューラは、スケジューラジョブを作成することで作成できます。このガイドの目的のために、`WandbScheduler`を変更してログを増やします。

1. `wandb/launch-jobs`リポジトリ（具体的には：`wandb/launch-jobs/jobs/sweep_schedulers`）をクローンする
2. より多くのログを提供するために`wandb_scheduler.py`を変更します。例：`_poll`関数にロギングを追加します。これは、新しいsweep runsを開始する前のポーリングサイクルごとに（設定可能なタイミングで）呼び出されます。
3. 変更されたファイルを使用してジョブを作成します：`python wandb_scheduler.py --project <project> --entity <entity> --name CustomWandbScheduler`
4. UIまたは前の呼び出しの出力で作成されたジョブの名前を特定します。特に指定がない限り、これはコードアーティファクトジョブになります。
5. スケジューラーが新しいジョブを指すようにsweep設定を作成します！

```yaml
...
scheduler:
  job: '<entity>/<project>/job-CustomWandbScheduler:latest'
...
```

  </TabItem>
  <TabItem value="optuna-scheduler">

Optunaは、特定のモデルに対して最適なハイパーパラメータを見つけるためにさまざまなアルゴリズムを使用するハイパーパラメータ最適化フレームワークです（W&Bと類似しています）。[サンプリングアルゴリズム](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html)に加えて、Optunaは[プルーニングアルゴリズム](https://optuna.readthedocs.io/en/stable/reference/pruners.html)も提供しており、パフォーマンスの悪いrunを早期に終了させることができます。これは、多くのrunを実行する際に時間とリソースを節約するのに特に有用です。クラスは非常に設定可能で、期待されるパラメータを`scheduler.settings.pruner/sampler.args`ブロックに渡すだけです。

Optunaのスケジューリングロジックをジョブとして使用してLaunch sweepを作成します。

1. まず、自分のジョブを作成するか、事前に作成されたOptunaスケジューラ画像のジョブを使用します。
    * 自分のジョブを作成する方法の例については、[`wandb/launch-jobs`](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers)リポジトリを参照してください。
    * 事前に作成されたOptuna画像を使用する場合、`wandb/sweep-jobs`プロジェクト内の`job-optuna-sweep-scheduler`に移動するか、ジョブ名を使用することができます：`wandb/sweep-jobs/job-optuna-sweep-scheduler:latest`。
    

2. ジョブを作成した後、sweepを作成できます。Optunaスケジューラジョブを指す`job`キーを含む`scheduler`ブロックを含むsweep設定を構築します（以下の例を参照）。

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

3. 最後に、launch-sweepコマンドを使用してアクティブなキューにsweepをローンチします：
  
```bash
wandb launch-sweep <config.yaml> -q <queue> -p <project> -e <entity>
```

Optunaスイープスケジューラジョブの具体的な実装については、[wandb/launch-jobs](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers/optuna_scheduler/optuna_scheduler.py)を参照してください。Optunaスケジューラで可能なことの例については、[wandb/examples](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/optuna-scheduler)をチェックしてください。


  </TabItem>
</Tabs>

カスタムsweepスケジューラジョブで可能な例は、`jobs/sweep_schedulers`ディレクトリにある[wandb/launch-jobs](https://github.com/wandb/launch-jobs)リポジトリにあります。このガイドでは、一般に公開されている**Wandb Scheduler Job**の使用方法と、カスタムsweepスケジューラジョブを作成するプロセスをご紹介します。


 ## Launchでsweepsを再開する方法
以前に起動されたsweepからLaunch-sweepを再開することも可能です。ハイパーパラメータやトレーニングジョブは変更できませんが、スケジューラスペシフィックなパラメータや送信先のキューは変更できます。

:::info
初期のsweepが "latest"のようなエイリアスを持つトレーニングジョブを使用していた場合、再開すると結果が異なる可能性があります。これは、最後のrun以降に最新のジョブバージョンが変更された場合です。
:::

  1. 以前に実行されたlaunch sweepのsweep名/IDを特定します。sweep IDは8文字の文字列（例えば、`hhd16935`）であり、W&B Appのプロジェクトで確認できます。
  2. スケジューラのパラメータを変更する場合は、更新された設定ファイルを構築します。
  3. ターミナルで、以下のコマンドを実行します。`<`と`>`で囲まれた内容を自分の情報で置き換えてください：

```bash
wandb launch-sweep <optional config.yaml> --resume_id <sweep id> --queue <queue_name>
```