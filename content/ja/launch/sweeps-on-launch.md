---
title: W&B Launch を使用してスイープを作成する
description: スイープのローンチでハイパーパラメータスイープを自動化する方法を発見する。
menu:
  launch:
    identifier: ja-launch-sweeps-on-launch
    parent: launch
url: guides/launch/sweeps-on-launch
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1WxLKaJlltThgZyhc7dcZhDQ6cjVQDfil#scrollTo=AFEzIxA6foC7" >}}

ハイパーパラメータチューニングジョブを W&B Launch を使って作成します。Launch の Sweeps を使用すると、指定されたハイパーパラメーターでスイープするためのスイープスケジューラーが、Launch Queue にプッシュされます。スイープスケジューラーはエージェントによってピックアップされると開始され、選択されたハイパーパラメーターでスイープする run を同じキューにローンチします。これはスイープが終了するか、または停止するまで続きます。

デフォルトの W&B Sweep スケジューリングエンジンを使用するか、独自のカスタムスケジューラーを実装することができます。

1. 標準スイープスケジューラー: デフォルトの W&B Sweep スケジューリングエンジンを使用して [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) を制御します。`bayes`、`grid`、`random` メソッドが利用可能です。
2. カスタムスイープスケジューラー: スイープスケジューラーをジョブとして動作するように設定します。このオプションにより、完全なカスタマイズが可能です。標準のスイープスケジューラーを拡張して追加のログを含める方法の例は、以下のセクションに記載されています。

{{% alert %}}
このガイドは、W&B Launch が事前に設定されていることを前提としています。W&B Launch が設定されていない場合は、Launch ドキュメントの [開始方法]({{< relref path="./#how-to-get-started" lang="ja" >}}) セクションを参照してください。
{{% /alert %}}

{{% alert %}}
初めて Launch で Sweeps を使用する場合は、「basic」メソッドを使用してスイープを作成することをお勧めします。W&B の標準スケジューリングエンジンがニーズを満たさない場合は、カスタムの Launch 用スイープスケジューラーを使用してください。
{{% /alert %}}

## W&B 標準スケジューラーを使用したスイープの作成
Launch で W&B Sweeps を作成します。W&B App を使用してインタラクティブにスイープを作成することも、W&B CLI を使用してプログラム的にスイープを作成することもできます。Launch スイープの高度な設定、スケジューラーのカスタマイズを可能にする CLI を使用します。

{{% alert %}}
W&B Launch でスイープを作成する前に、まずスイープするジョブを作成しておいてください。[Create a Job]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ja" >}}) ページを参照してください。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B app" %}}

W&B App を使って、インタラクティブにスイープを作成します。

1. W&B App であなたの W&B プロジェクトへ移動します。
2. 左側のパネルからスイープのアイコン（ほうきのイメージ）を選択します。
3. 次に、**Create Sweep** ボタンを選択します。
4. **Configure Launch 🚀** ボタンをクリックします。
5. **Job** ドロップダウンメニューから、スイープを作成したい仕事の名前とバージョンを選択します。
6. **Queue** ドロップダウンメニューを使用して、スイープを実行するキューを選択します。
7. **Job Priority** ドロップダウンで、Launch ジョブの優先度を指定します。Launch Queue が優先度をサポートしていない場合、Launch ジョブの優先度は「Medium」に設定されます。
8. (オプション) run またはスイープスケジューラーの引数を override 設定します。例えば、スケジューラーオーバーライドを使用して、スケジューラーが管理する同時 run の数を `num_workers` を使って設定します。
9. (オプション) **Destination Project** ドロップダウンメニューを使用してスイープを保存するプロジェクトを選択します。
10. **Save** をクリックします。
11. **Launch Sweep** を選択します。

{{< img src="/images/launch/create_sweep_with_launch.png" alt="" >}}

{{% /tab %}}
{{% tab "CLI" %}}

W&B CLI を使って、Launch でプログラム的に W&B Sweep を作成します。

1. Sweep の設定を作成します。
2. スイープ設定内で完全なジョブ名を指定します。
3. スイープエージェントを初期化します。

{{% alert %}}
ステップ 1 と 3 は、通常 W&B Sweep を作成するときに行うステップと同じです。
{{% /alert %}}

例えば、以下のコードスニペットでは、ジョブの値に `'wandb/jobs/Hello World 2:latest'` を指定しています。

```yaml
# launch-sweep-config.yaml

job: 'wandb/jobs/Hello World 2:latest'
description: launch jobsを使ったスイープの例

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

# スケジューラ用のオプションパラメーター:

# scheduler:
#   num_workers: 1  # 同時にスイープを実行するスレッド数
#   docker_image: <スケジューラー用のベースイメージ>
#   resource: <例: local-container...>
#   resource_args:  # run に渡されるリソース引数
#     env: 
#         - WANDB_API_KEY

# Launch 用のオプションパラメーター
# launch: 
#    registry: <イメージのプル用のレジストリ>
```

スイープ設定の作成方法についての情報は、[Define sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration.md" lang="ja" >}}) ページを参照してください。

4. 次に、スイープを初期化します。設定ファイルのパス、ジョブキューの名前、W&B エンティティ、プロジェクトの名前を指定します。

```bash
wandb launch-sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity> --project <project_name>
```

W&B Sweeps についての詳細は、[Tune Hyperparameters]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) チャプターを参照してください。

{{% /tab %}}
{{< /tabpane >}}


## カスタムスイープスケジューラーの作成
W&B スケジューラーまたはカスタムスケジューラーを使用してカスタムスイープスケジューラーを作成します。

{{% alert %}}
スケジューラージョブを使用するには、wandb cli バージョンが `0.15.4` 以上である必要があります。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B scheduler" %}}
  W&B スイープスケジューリングロジックをジョブとして使用して Launch スイープを作成します。
  
  1. パブリックの wandb/sweep-jobs プロジェクトで Wandb スケジューラージョブを識別するか、以下のジョブ名を使用します: `'wandb/sweep-jobs/job-wandb-sweep-scheduler:latest'`
  2. この名前を指す `job` キーを含む追加の `scheduler` ブロックで設定 yaml を構築します。例は以下の通りです。
  3. 新しい設定で `wandb launch-sweep` コマンドを使用します。

例の設定:
```yaml
# launch-sweep-config.yaml  
description: スケジューラージョブを使用したLaunchスイープ設定
scheduler:
  job: wandb/sweep-jobs/job-wandb-sweep-scheduler:latest
  num_workers: 8  # 8つの同時スイープ実行を許可

# スイープが実行するトレーニング/チューニングジョブ
job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
method: grid
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
```
{{% /tab %}}
{{% tab "カスタムスケジューラー" %}}
  カスタムスケジューラーは、スケジューラージョブを作成することで作成できます。このガイドの目的のために、`WandbScheduler` を変更してより多くのログを提供します。

  1. `wandb/launch-jobs` リポジトリをクローンします（特定の場所: `wandb/launch-jobs/jobs/sweep_schedulers`）
  2. これで、`wandb_scheduler.py` を修正して、ログの増加を達成できます。例: `_poll` 関数にログを追加します。これは、毎回のポーリングサイクル（設定可能なタイミング）で呼び出され、次のスイープ run をローンチする前に行います。
  3. 修正したファイルを実行して、以下のコマンドでジョブを作成します: `python wandb_scheduler.py --project <project> --entity <entity> --name CustomWandbScheduler`
  4. 作成されたジョブの名前を、UI または前の呼び出しの出力で識別します。特に指定がない場合、これはコードアーティファクトジョブです。
  5. スケジューラーが新しいジョブを指すようにスイープの設定を作成します。

```yaml
...
scheduler:
  job: '<entity>/<project>/job-CustomWandbScheduler:latest'
...
```

{{% /tab %}}
{{% tab "Optuna スケジューラー" %}}

  Optuna は、指定されたモデルに最適なハイパーパラメーターを見つけるために様々なアルゴリズムを使用するハイパーパラメータ最適化フレームワークです（W&B と似ています）。[サンプリングアルゴリズム](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) に加え、Optuna は数多くの [プルーニングアルゴリズム](https://optuna.readthedocs.io/en/stable/reference/pruners.html) も提供し、パフォーマンスが低い run を早期に終了させることができます。多数の run を実行する際、これは時間とリソースを節約するのに特に有用です。クラスは非常に設定可能で、`scheduler.settings.pruner/sampler.args` ブロックに期待されるパラメーターを渡すだけです。

Optuna のスケジューリングロジックをジョブとして使用して Launch スイープを作成します。

1. まず、独自のジョブを作成するか、ビルド済みの Optuna スケジューラーイメージジョブを使用します。
    * 独自のジョブを作成する方法の例については、[`wandb/launch-jobs`](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers) リポジトリを参照してください。
    * ビルド済みの Optuna イメージを使用するには、`wandb/sweep-jobs` プロジェクトで `job-optuna-sweep-scheduler` へ移動するか、ジョブ名を使用します: `wandb/sweep-jobs/job-optuna-sweep-scheduler:latest`。

2. ジョブを作成した後、スイープを作成できます。Optuna スケジューラージョブを指す `job` キーを含む `scheduler` ブロックを含むスイープ設定を構築します（以下の例）。

```yaml
  # optuna_config_basic.yaml
  description: ベーシックな Optuna スケジューラー
  job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
  run_cap: 5
  metric:
    name: epoch/val_loss
    goal: minimize

  scheduler:
    job: wandb/sweep-jobs/job-optuna-sweep-scheduler:latest
    resource: local-container  # イメージからソースされるスケジューラージョブに必須
    num_workers: 2

    # optuna 特有の設定
    settings:
      pruner:
        type: PercentilePruner
        args:
          percentile: 25.0  # 75% の run を終了
          n_warmup_steps: 10  # 最初の x ステップではプルーニングを無効に

  parameters:
    learning_rate:
      min: 0.0001
      max: 0.1
  ```

  3. 最後に、launch-sweep コマンドでアクティブキューにスイープをローンチします。

  ```bash
  wandb launch-sweep <config.yaml> -q <queue> -p <project> -e <entity>
  ```

  Optuna スイープスケジューラージョブの正確な実装については、[wandb/launch-jobs](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers/optuna_scheduler/optuna_scheduler.py) を参照してください。Optuna スケジューラーで可能な例については、[wandb/examples](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/optuna-scheduler) をチェックしてください。
{{% /tab %}}
{{< /tabpane >}}

カスタムスイープスケジューラージョブで可能な例は、`jobs/sweep_schedulers` 以下の [wandb/launch-jobs](https://github.com/wandb/launch-jobs) リポジトリにあります。このガイドは、公開されている **Wandb スケジューラージョブ** の使用方法を示し、またカスタムスイープスケジューラージョブを作成するプロセスを示しています。


## launch で sweeps を再開する方法
以前に実行されたスイープから launch-sweep を再開することも可能です。ハイパーパラメーターとトレーニングジョブは変更できませんが、スケジューラー固有のパラメーターおよび送信先キューは変更可能です。

{{% alert %}}
最初のスイープが「latest」などのエイリアスを持つトレーニングジョブを使用していた場合、再開すると最新のジョブバージョンが最後に実行されたジョブと変わっている場合に異なる結果をもたらす可能性があります。
{{% /alert %}}

1. 以前に実行された launch-sweep のスイープ名/ID を特定します。スイープ ID は 8 文字の文字列です（例：`hhd16935`）。W&B App のプロジェクトで見つけることができます。
2. スケジューラーパラメーターを変更する場合、更新された設定ファイルを構成します。
3. ターミナルで、次のコマンドを実行します。`<` と `>` で囲まれている内容を自身の情報で置き換えます。

```bash
wandb launch-sweep <optional config.yaml> --resume_id <sweep id> --queue <queue_name>
```