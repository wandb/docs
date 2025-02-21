---
title: Create sweeps with W&B Launch
description: ローンンチでハイパーパラメータ スイープを自動化する方法を見つけましょう。
menu:
  launch:
    identifier: ja-launch-sweeps-on-launch
    parent: launch
url: guides/launch/sweeps-on-launch
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1WxLKaJlltThgZyhc7dcZhDQ6cjVQDfil#scrollTo=AFEzIxA6foC7" >}}

W&B Launch を使ってハイパーパラメータチューニングジョブ（[sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}})）を作成します。Launch 上での Sweeps では、指定されたハイパーパラメータでスイープするためにスイープスケジューラーが Launch Queue にプッシュされます。スイープスケジューラーはエージェントにより取得され次第開始され、選択されたハイパーパラメータで同じキューにスイープされる run をローンチします。これはスイープが終了するか、停止されるまで続きます。

デフォルトの W&B Sweep スケジューリングエンジンを使用するか、独自のカスタムスケジューラーを実装することができます。

1. 標準スイープスケジューラー: [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) を管理するデフォルトの W&B Sweep スケジューリングエンジンを使用します。おなじみの `bayes`、`grid`、`random` のメソッドが利用可能です。
2. カスタムスイープスケジューラー: スイープスケジューラーをジョブとして実行するように設定します。このオプションでは全てのカスタマイズが可能です。標準のスイープスケジューラーを拡張して、より多くのログを含むようにする方法の例は、下のセクションで見つけることができます。

{{% alert %}}
このガイドは事前に W&B Launch が設定されていることを前提としています。W&B Launch が設定されていない場合は、ローンチドキュメントの [開始方法]({{< relref path="./#how-to-get-started" lang="ja" >}}) セクションを参照してください。
{{% /alert %}}

{{% alert %}}
W&B Launch で初めて Sweeps を使用する場合は「基本」メソッドでスイープを作成することをお勧めします。標準の W&B スケジューリングエンジンがニーズに合わない場合は、カスタムスイープスケジューラーを使用してください。
{{% /alert %}}

## W&B 標準スケジューラーでスイープを作成する
Launch を使って W&B Sweeps を作成します。W&B App でインタラクティブにスイープを作成することも、W&B CLI を使ってプログラム的に作成することもできます。スケジューラーをカスタマイズする能力を含む Launch sweeps の高度な設定には、CLI を使用します。

{{% alert %}}
W&B Launch を使用してスイープを作成する前に、スイープするジョブを最初に作成することを確認してください。[Create a Job]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ja" >}}) ページに詳細があります。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B app" %}}

W&B App を使ってインタラクティブにスイープを作成します。

1. W&B App 内の W&B プロジェクトに移動します。  
2. 左パネルのスイープアイコン（ほうきのイメージ）を選択します。
3. 次に、**Create Sweep** ボタンを選択します。
4. **Configure Launch 🚀** ボタンをクリックします。
5. **Job** ドロップダウンメニューから、スイープを作成したいジョブとそのバージョンを選択します。
6. **Queue** ドロップダウンメニューを使用してスイープを実行するキューを選択します。
8. **Job Priority** ドロップダウンを使用してローンチジョブの優先度を指定します。ローンチキューが優先度をサポートしていない場合、ローンチジョブの優先度は「Medium」に設定されます。
8. （オプション）run またはスイープスケジューラーのオーバーライド引数を設定します。例えば、スケジューラーのオーバーライドを使用して、スケジューラーが管理する同時実行 run の数を `num_workers` を使って設定します。
9. （オプション）**Destination Project** ドロップダウンメニューを使用して、スイープを保存するプロジェクトを選択します。
10. **Save** をクリックします。
11. **Launch Sweep** を選択します。

{{< img src="/images/launch/create_sweep_with_launch.png" alt="" >}}

{{% /tab %}}
{{% tab "CLI" %}}

W&B CLI を使用してプログラム的に W&B Sweep を Launch で作成します。

1. スイープ設定を作成します。
2. スイープ設定内に完全なジョブ名を指定します。
3. スイープエージェントを初期化します。

{{% alert %}}
ステップ 1 と 3 は、通常 W&B Sweep を作成するときに行う手順と同じです。
{{% /alert %}}

例えば、以下のコードスニペットではジョブ値として `'wandb/jobs/Hello World 2:latest'` を指定します：

```yaml
# launch-sweep-config.yaml

job: 'wandb/jobs/Hello World 2:latest'
description: launch jobs を使用したスイープ例

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

# スケジューラのオプションパラメータ:

# scheduler:
#   num_workers: 1  # 同時に実行するスイープ run の数
#   docker_image: <スケジューラーのベースイメージ>
#   resource: <例. local-container...>
#   resource_args:  # run に渡すリソース引数
#     env: 
#         - WANDB_API_KEY

# Launch のオプションパラメータ
# launch: 
#    registry: <イメージ取得用レジストリ>
```

スイープ設定の作成方法の詳細については、[スイープ設定の定義]({{< relref path="/guides/models/sweeps/define-sweep-configuration.md" lang="ja" >}}) ページを参照してください。

4. 次にスイープを初期化します。設定ファイルのパス、ジョブキューの名前、W&B エンティティ、プロジェクトの名前を指定します。

```bash
wandb launch-sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>
```

W&B Sweeps についての詳細は、[ハイパーパラメータのチューニング]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) チャプターを参照してください。

{{% /tab %}}
{{< /tabpane >}}

## カスタムスイープスケジューラーを作成する
W&B スケジューラーまたはカスタムスケジューラーを使ってカスタムスイープスケジューラーを作成します。

{{% alert %}}
スケジューラージョブを使用するには wandb cli バージョンが `0.15.4` 以上である必要があります。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B scheduler" %}}
  W&B スイープスケジューリングロジックを使用してスイープをジョブとして開始します。

  1. 公開されている wandb/sweep-jobs プロジェクト内の Wandb スケジューラージョブを特定するか、以下のジョブ名を使用します：
  `'wandb/sweep-jobs/job-wandb-sweep-scheduler:latest'`
  2. `job` キーがこの名前を指す `scheduler` ブロックを含む設定 yaml を構築します。以下に例を示します。
  3. 新しい設定で `wandb launch-sweep` コマンドを使用します。

設定例：
```yaml
# launch-sweep-config.yaml  
description: スケジューラージョブを使用したスイープ設定のの起動
scheduler:
  job: wandb/sweep-jobs/job-wandb-sweep-scheduler:latest
  num_workers: 8  # 同時に実行するスイープ run の数

# スィープ run が実行するトレーニング／チューニングジョブ
job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
method: grid
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
```
{{% /tab %}}
{{% tab "Custom scheduler" %}}
  `WandbScheduler` を修正して、より多くのログを表示するようにカスタムスケジューラーを作成することができます。

  1. `wandb/launch-jobs` リポジトリをクローンします（特に：`wandb/launch-jobs/jobs/sweep_schedulers`）
  2. `wandb_scheduler.py` を修正して、より多くのログを取得します。例：関数 `_poll` にログを追加します。これは、新しいスイープ run を開始する前に、各ポーリングサイクル（設定可能）ごとに呼び出されます。
  3. 次のコマンドでジョブを作成するように修正ファイルを実行します：`python wandb_scheduler.py --project <project> --entity <entity> --name CustomWandbScheduler`
  4. UI または前の呼び出しの出力で作成されたジョブの名前を確認します。特に指定しない限り、これはコードアーティファクトジョブとなるでしょう。
  5. スケジューラーが新しいジョブを指すスイープ設定を作成します。

```yaml
...
scheduler:
  job: '<entity>/<project>/job-CustomWandbScheduler:latest'
...
```

{{% /tab %}}
{{% tab "Optuna scheduler" %}}

  Optuna は、特定のモデルに対して最適なハイパーパラメータを見つけるためにさまざまなアルゴリズムを使用するハイパーパラメータ最適化フレームワークです（W&B と類似しています）。[サンプリングアルゴリズム](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) に加えて、Optuna は指定されたパラメータで不良な run を早期に終了するための多様な [プルーニングアルゴリズム](https://optuna.readthedocs.io/en/stable/reference/pruners.html) も提供します。これは多数の run を実行する際に特に有用であり、時間と資源の節約に役立ちます。クラスは非常に設定可能であり、構成ファイルの `scheduler.settings.pruner/sampler.args` ブロックに期待されるパラメータを渡すだけで済みます。

Optuna のスケジューリングロジックを使用してスイープをジョブとして開始します。

1. まず、自分自身のジョブを作成するか、事前に作成された Optuna スケジューラーイメージジョブを使用します。
   * 自身のジョブの作成方法についての例は、[`wandb/launch-jobs`](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers) リポジトリを参照してください。
   * 事前に作成された Optuna イメージを使用するには、`wandb/sweep-jobs` プロジェクト内の `job-optuna-sweep-scheduler` に移動するか、以下のジョブ名を使用できます：`wandb/sweep-jobs/job-optuna-sweep-scheduler:latest`。
   
2. ジョブを作成した後、スイープを作成します。`scheduler` ブロックを含むスイープ設定を構築し、`job` キーが Optuna スケジューラージョブを指していることを確認します（例を以下に示します）。

```yaml
  # optuna_config_basic.yaml
  description: 基本的な Optuna スケジューラー
  job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
  run_cap: 5
  metric:
    name: epoch/val_loss
    goal: minimize

  scheduler:
    job: wandb/sweep-jobs/job-optuna-sweep-scheduler:latest
    resource: local-container  # イメージからソースされるスケジューラージョブに必要
    num_workers: 2

    # optuna 固有の設定
    settings:
      pruner:
        type: PercentilePruner
        args:
          percentile: 25.0  # 75% の run を停止
          n_warmup_steps: 10  # 最初の x ステップはプルーニングをオフ

  parameters:
    learning_rate:
      min: 0.0001
      max: 0.1
  ```

3. 最後に、launch-sweep コマンドを使用してスイープをアクティブなキューにローンチします：
  
  ```bash
  wandb launch-sweep <config.yaml> -q <queue> -p <project> -e <entity>
  ```

Optuna スイープスケジューラージョブの正確な実装については、[wandb/launch-jobs](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers/optuna_scheduler/optuna_scheduler.py) を参照してください。Optuna スケジューラーで何が可能かについてのさらなる例は、[wandb/examples](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/optuna-scheduler) をチェックしてください。
{{% /tab %}}
{{< /tabpane >}}

 カスタムスイープスケジューラージョブで可能なことの例は、`jobs/sweep_schedulers` の下にある [wandb/launch-jobs](https://github.com/wandb/launch-jobs) リポジトリで利用可能です。このガイドは、公開されている **Wandb Scheduler Job** を使用する方法を示していますが、カスタムスイープスケジューラージョブを作成するプロセスをも示しています。

## Launch 上のスイープを再開する方法
  以前に開始されたスイープから launch-sweep を再開することも可能です。ハイパーパラメータやトレーニングジョブは変更できませんが、スケジューラー固有のパラメータやプッシュ先のキューは変更可能です。

{{% alert %}}
最初のスイープで 'latest' などのエイリアスを持つトレーニングジョブを使用した場合、ジョブの最新バージョンが前回の run 以降に変更されている場合、再開すると異なる結果が得られることがあります。
{{% /alert %}}

1. 以前に実行された launch-sweep のスイープ名/ID を特定します。スイープ ID は、W&B App 内のプロジェクトで見つけることができる8桁の文字列（例：`hhd16935`）です。
2. スケジューラーパラメータを変更する場合は、更新された設定ファイルを構築します。
3. ターミナルで次のコマンドを実行し、`<` と `>` で囲まれた内容をあなたの情報に置き換えます： 

```bash
wandb launch-sweep <optional config.yaml> --resume_id <sweep id> --queue <queue_name>
```