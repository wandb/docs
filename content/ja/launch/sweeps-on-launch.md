---
title: Create sweeps with W&B Launch
description: Launch でハイパーパラメータの Sweeps を自動化する方法をご覧ください。
menu:
  launch:
    identifier: ja-launch-sweeps-on-launch
    parent: launch
url: guides/launch/sweeps-on-launch
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1WxLKaJlltThgZyhc7dcZhDQ6cjVQDfil#scrollTo=AFEzIxA6foC7" >}}

W&B Launch でハイパーパラメータチューニングジョブ ( [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) ) を作成します。 Launch で Sweeps を行うと、sweep スケジューラは、sweep するために指定されたハイパーパラメータとともに Launch Queue にプッシュされます。 sweep スケジューラは、エージェントによって選択されると開始され、選択されたハイパーパラメータを使用して sweep の run を同じキューに起動します。 これは、sweep が終了または停止するまで継続されます。

デフォルトの W&B Sweep スケジューリングエンジンを使用するか、独自のカスタムスケジューラを実装できます。

1. 標準 sweep スケジューラ: [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) を制御するデフォルトの W&B Sweep スケジューリングエンジンを使用します。 使い慣れた `bayes`、`grid`、および `random` の method を使用できます。
2. カスタム sweep スケジューラ: sweep スケジューラをジョブとして実行するように設定します。 このオプションを使用すると、完全にカスタマイズできます。 標準の sweep スケジューラを拡張して、より多くのログを含める方法の例を以下のセクションに示します。

{{% alert %}}
このガイドでは、W&B Launch が以前に設定されていることを前提としています。 W&B Launch が設定されていない場合は、Launch ドキュメントの [開始方法]({{< relref path="./#how-to-get-started" lang="ja" >}}) セクションを参照してください。
{{% /alert %}}

{{% alert %}}
Launch で Sweeps を初めて使用する場合は、「basic」method を使用して Launch で sweep を作成することをお勧めします。 標準の W&B スケジューリングエンジンがニーズを満たさない場合は、Launch スケジューラでカスタム Sweeps を使用します。
{{% /alert %}}

## W&B 標準スケジューラで sweep を作成する
Launch で W&B Sweeps を作成します。 W&B App でインタラクティブに sweep を作成するか、W&B CLI でプログラムで作成できます。 スケジューラのカスタマイズ機能など、Launch Sweeps の高度な設定を行うには、CLI を使用します。

{{% alert %}}
W&B Launch で sweep を作成する前に、まず sweep するジョブを作成してください。 詳細については、[ジョブの作成]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ja" >}}) ページを参照してください。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B app" %}}

W&B App でインタラクティブに sweep を作成します。

1. W&B App で W&B プロジェクトに移動します。
2. 左側のパネルにある sweeps アイコン (ほうきの画像) を選択します。
3. 次に、**Sweep の作成** ボタンを選択します。
4. **Launch の設定🚀** ボタンをクリックします。
5. **ジョブ** ドロップダウンメニューから、ジョブの名前と sweep の作成元となるジョブの バージョンを選択します。
6. **キュー** ドロップダウンメニューを使用して、sweep を実行するキューを選択します。
7. **ジョブの優先度** ドロップダウンを使用して、Launch ジョブの優先度を指定します。 Launch キューが優先順位付けをサポートしていない場合、Launch ジョブの優先度は「中」に設定されます。
8. (オプション) run または sweep スケジューラのオーバーライド arg を設定します。 たとえば、スケジューラのオーバーライドを使用して、`num_workers` を使用してスケジューラが管理する同時実行 run の数を設定します。
9. (オプション) **宛先プロジェクト** ドロップダウンメニューを使用して、sweep を保存するプロジェクトを選択します。
10. **保存** をクリックします。
11. **Sweep の起動** を選択します。

{{< img src="/images/launch/create_sweep_with_launch.png" alt="" >}}

{{% /tab %}}
{{% tab "CLI" %}}

W&B CLI を使用して、プログラムで Launch で W&B Sweep を作成します。

1. Sweep 設定を作成します。
2. sweep 設定内でジョブの完全な名前を指定します。
3. sweep agent を初期化します。

{{% alert %}}
手順 1 と 3 は、W&B Sweep を作成するときに通常行う手順と同じです。
{{% /alert %}}

たとえば、次の コードスニペット では、ジョブの値に `'wandb/jobs/Hello World 2:latest'` を指定します。

```yaml
# launch-sweep-config.yaml

job: 'wandb/jobs/Hello World 2:latest'
description: launch jobs を使用した sweep の例

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

sweep 設定の作成方法については、[sweep 設定の定義]({{< relref path="/guides/models/sweeps/define-sweep-configuration.md" lang="ja" >}}) ページを参照してください。

4. 次に、sweep を初期化します。 設定ファイルへのパス、ジョブキューの名前、W&B エンティティ、およびプロジェクトの名前を指定します。

```bash
wandb launch-sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>
```

W&B Sweeps の詳細については、[ハイパーパラメータの チューニング]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) のチャプターを参照してください。

{{% /tab %}}
{{< /tabpane >}}

## カスタム sweep スケジューラを作成する
W&B スケジューラまたはカスタムスケジューラのいずれかを使用して、カスタム sweep スケジューラを作成します。

{{% alert %}}
スケジューラジョブを使用するには、wandb cli バージョン >= `0.15.4` が必要です。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B scheduler" %}}
  W&B sweep スケジューリングロジックをジョブとして使用して、Launch sweep を作成します。

  1. パブリック wandb/sweep-jobs プロジェクトで Wandb スケジューラジョブを識別するか、ジョブ名を使用します。
  `'wandb/sweep-jobs/job-wandb-sweep-scheduler:latest'`
  2. この名前を指す `job` キーを含む追加の `scheduler` ブロックを使用して、構成 yaml を作成します。以下の例を参照してください。
  3. 新しい設定で `wandb launch-sweep` コマンドを使用します。

構成例:
```yaml
# launch-sweep-config.yaml
description: スケジューラジョブを使用した Launch sweep 構成
scheduler:
  job: wandb/sweep-jobs/job-wandb-sweep-scheduler:latest
  num_workers: 8  # 8 つの同時 sweep run を許可します

# sweep runs が実行するトレーニング/チューニングジョブ
job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
method: grid
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
```
{{% /tab %}}
{{% tab "Custom scheduler" %}}
  カスタムスケジューラは、スケジューラジョブを作成することで作成できます。 このガイドの目的のために、ログをより多く提供するために `WandbScheduler` を変更します。

  1. `wandb/launch-jobs` リポジトリ (具体的には `wandb/launch-jobs/jobs/sweep_schedulers`) を複製します。
  2. これで、`wandb_scheduler.py` を変更して、必要なログの増加を実現できます。 例: 関数 `_poll` にログを追加します。 これは、新しい sweep run を起動する前に、ポーリングサイクルごとに 1 回 (設定可能なタイミング) 呼び出されます。
  3. 変更したファイルを実行してジョブを作成します。`python wandb_scheduler.py --project <project> --entity <entity> --name CustomWandbScheduler`
  4. UI または前の呼び出しの出力で、作成されたジョブの名前を特定します。これは、コードアーティファクトジョブになります (特に指定されていない場合)。
  5. 次に、スケジューラが新しいジョブを指す sweep 構成を作成します。

```yaml
...
scheduler:
  job: '<entity>/<project>/job-CustomWandbScheduler:latest'
...
```

{{% /tab %}}
{{% tab "Optuna scheduler" %}}

  Optuna は、特定のモデルに最適なハイパーパラメータを見つけるためにさまざまなアルゴリズムを使用するハイパーパラメータ最適化 フレームワーク です (W&B と同様)。 Optuna は、[サンプリングアルゴリズム](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html)に加えて、パフォーマンスの低い run を早期に終了するために使用できるさまざまな[枝刈りアルゴリズム](https://optuna.readthedocs.io/en/stable/reference/pruners.html)も提供します。 これは、多数の run を実行する場合に特に役立ちます。時間とリソースを節約できるためです。 クラスは高度に設定可能で、設定ファイルの `scheduler.settings.pruner/sampler.args` ブロックに必要なパラメータを渡すだけです。

Optuna のスケジューリングロジックをジョブで使用して、Launch sweep を作成します。

1. まず、独自のジョブを作成するか、事前構築済みの Optuna スケジューライメージジョブを使用します。
    * 独自のジョブを作成する方法の例については、[`wandb/launch-jobs`](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers) リポジトリを参照してください。
    * 事前構築済みの Optuna イメージを使用するには、`wandb/sweep-jobs` プロジェクトの `job-optuna-sweep-scheduler` に移動するか、ジョブ名 `wandb/sweep-jobs/job-optuna-sweep-scheduler:latest` を使用できます。

2. ジョブを作成したら、sweep を作成できます。 Optuna スケジューラジョブを指す `job` キーを持つ `scheduler` ブロックを含む sweep 構成を作成します (以下の例)。

```yaml
  # optuna_config_basic.yaml
  description: 基本的な Optuna スケジューラ
  job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
  run_cap: 5
  metric:
    name: epoch/val_loss
    goal: minimize

  scheduler:
    job: wandb/sweep-jobs/job-optuna-sweep-scheduler:latest
    resource: local-container  # イメージから取得したスケジューラジョブに必要
    num_workers: 2

    # optuna 固有の設定
    settings:
      pruner:
        type: PercentilePruner
        args:
          percentile: 25.0  # run の 75% を強制終了
          n_warmup_steps: 10  # 最初の x ステップでは枝刈りはオフ

  parameters:
    learning_rate:
      min: 0.0001
      max: 0.1
  ```

  3. 最後に、launch-sweep コマンドを使用して、アクティブなキューに sweep を起動します。

  ```bash
  wandb launch-sweep <config.yaml> -q <queue> -p <project> -e <entity>
  ```

  Optuna sweep スケジューラジョブの正確な実装については、[wandb/launch-jobs](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers/optuna_scheduler/optuna_scheduler.py) を参照してください。 Optuna スケジューラで可能なことのより多くの例については、[wandb/examples](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/optuna-scheduler) を確認してください。
{{% /tab %}}
{{< /tabpane >}}

カスタム sweep スケジューラジョブで可能なことの例は、`jobs/sweep_schedulers` の [wandb/launch-jobs](https://github.com/wandb/launch-jobs) リポジトリにあります。 このガイドでは、一般公開されている **Wandb スケジューラジョブ** の使用方法と、カスタム sweep スケジューラジョブを作成するプロセスについて説明します。

## Launch で sweep を再開する方法
  以前に Launch された sweep から Launch sweep を再開することも可能です。 ハイパーパラメータとトレーニングジョブは変更できませんが、スケジューラ固有のパラメータとプッシュ先のキューは変更できます。

{{% alert %}}
最初の sweep で「latest」のようなエイリアスを持つトレーニングジョブを使用した場合、最後の run 以降に最新のジョブバージョンが変更されていると、再開すると異なる結果になる可能性があります。
{{% /alert %}}

  1. 以前に実行された Launch sweep の sweep 名/ID を特定します。 sweep ID は 8 文字の文字列 (たとえば、`hhd16935`) で、W&B App のプロジェクトにあります。
  2. スケジューラのパラメータを変更する場合は、更新された構成ファイルを作成します。
  3. ターミナルで、次のコマンドを実行します。 `<` および `>` で囲まれたコンテンツを情報に置き換えます。

```bash
wandb launch-sweep <optional config.yaml> --resume_id <sweep id> --queue <queue_name>
```
