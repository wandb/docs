---
title: W&B Launch でスイープを作成する
description: ローンチでハイパーパラメータ sweep を自動化する方法をご紹介します。
menu:
  launch:
    identifier: sweeps-on-launch
    parent: launch
url: guides/launch/sweeps-on-launch
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1WxLKaJlltThgZyhc7dcZhDQ6cjVQDfil#scrollTo=AFEzIxA6foC7" >}}

W&B Launch を使ってハイパーパラメータチューニングジョブ（[sweeps]({{< relref "/guides/models/sweeps/" >}})）を作成しましょう。Launch 上で sweep を行うと、指定したハイパーパラメータを sweep する sweep スケジューラが Launch Queue に登録されます。スケジューラはエージェントによって取り上げられると実行を開始し、選択したハイパーパラメータで sweep run を同じキュー上で次々に開始します。これは sweep が終了、または停止されるまで続きます。

標準の W&B Sweep スケジューリングエンジンを使用するか、独自のカスタムスケジューラも実装できます。

1. 標準スイープスケジューラ: デフォルトの W&B Sweep スケジューリングエンジンを利用し、[W&B Sweeps]({{< relref "/guides/models/sweeps/" >}}) をコントロールします。おなじみの `bayes`、`grid`、`random` メソッドが利用可能です。
2. カスタムスイープスケジューラ: sweep スケジューラをジョブとして実行でき、完全なカスタマイズが可能です。標準のスイープスケジューラを拡張してロギングを追加する例は後述のセクションをご覧ください。

{{% alert %}}
このガイドは W&B Launch のセットアップが完了していることを前提としています。未設定の場合は、Launch ドキュメントの [開始方法]({{< relref "./#how-to-get-started" >}}) セクションをご参照ください。
{{% /alert %}}

{{% alert %}}
sweeps や launch を初めて利用する場合は、まず 'basic' メソッドで sweep を作成することをおすすめします。標準の W&B スケジューリングエンジンが要件を満たさない場合のみ、カスタムスケジューラをご利用ください。
{{% /alert %}}

## W&B 標準スケジューラで sweep を作成する
Launch で W&B Sweeps を作成します。sweep は W&B App からインタラクティブに作成するか、W&B CLI からプログラム的に作成できます。スケジューラのカスタマイズなど高度な設定が必要な場合には CLI をお使いください。

{{% alert %}}
W&B Launch で sweep を作成する前に、sweep の対象となるジョブを作成してください。詳細は [Create a Job]({{< relref "./create-and-deploy-jobs/create-launch-job.md" >}}) ページをご覧ください。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B app" %}}

W&B App を使ってインタラクティブに sweep を作成します。

1. W&B App で自分の W&B Project にアクセスします。  
2. 左側パネルで broom（ほうき）アイコンの sweeps を選択します。
3. **Create Sweep** ボタンをクリックします。
4. **Configure Launch** ボタンをクリックします。
5. **Job** ドロップダウンから sweep 対象のジョブ名とバージョンを選択します。
6. **Queue** ドロップダウンで、run を実行するキューを選択します。
7. **Job Priority** ドロップダウンで launch ジョブの優先度を指定します。launch queue が優先度設定に対応していなければ"Medium"となります。
8. （オプション）run や sweep スケジューラの引数を上書きできます。例えば scheduler の override で、`num_workers` を設定して同時に管理する run 数を指定できます。
9. （オプション）**Destination Project** ドロップダウンで sweep を保存する Project を選択します。
10. **Save** をクリックします。
11. **Launch Sweep** を選択します。

{{< img src="/images/launch/create_sweep_with_launch.png" alt="Launch sweep configuration" >}}

{{% /tab %}}
{{% tab "CLI" %}}

W&B CLI を使ってプログラム的に W&B Sweep を Launch で作成します。

1. Sweep 設定ファイルを作成
2. sweep 設定内でフルジョブ名を指定
3. sweep エージェントを初期化

{{% alert %}}
手順1と3は通常の W&B Sweep 作成時と同じです。
{{% /alert %}}

例えば、以下のコードスニペットではジョブ値に `'wandb/jobs/Hello World 2:latest'` を指定しています。

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

# オプションのスケジューラパラメータ:

# scheduler:
#   num_workers: 1  # 同時実行する sweep run の数
#   docker_image: <スケジューラ用ベースイメージ>
#   resource: <例: local-container など>
#   resource_args:  # run に渡すリソース引数
#     env: 
#         - WANDB_API_KEY

# オプションの Launch パラメータ
# launch: 
#    registry: <イメージプル用レジストリ>
```

sweep 設定ファイルの作成方法は [Define sweep configuration]({{< relref "/guides/models/sweeps/define-sweep-configuration.md" >}}) をご参照ください。

4. 次に sweep を初期化します。設定ファイルパス、ジョブキュー名、W&B Entity、Project 名を指定します。

```bash
wandb launch-sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>
```

W&B Sweeps の詳細は [Tune Hyperparameters]({{< relref "/guides/models/sweeps/" >}}) チャプターをご覧ください。

{{% /tab %}}
{{< /tabpane >}}

## カスタム sweep スケジューラを作成する
W&B スケジューラやカスタムスケジューラを使って独自の sweep スケジューラを作成できます。

{{% alert %}}
scheduler ジョブの利用には wandb cli バージョンが `0.15.4` 以上必要です
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B scheduler" %}}
  W&B sweep スケジューリングロジックをジョブとして使い、launch sweep を作成します。
  
  1. 公開プロジェクト wandb/sweep-jobs の Wandb scheduler ジョブを確認するか、ジョブ名 `'wandb/sweep-jobs/job-wandb-sweep-scheduler:latest'` を利用します。
  2. 追加の `scheduler` ブロックを持つ設定 yaml を作成し、`job` キーにこのジョブ名を指定します（例は下記）。
  3. 新しい設定ファイルで `wandb launch-sweep` コマンドを使います。

例:
```yaml
# launch-sweep-config.yaml  
description: Launch sweep config using a scheduler job
scheduler:
  job: wandb/sweep-jobs/job-wandb-sweep-scheduler:latest
  num_workers: 8  # 最大8つの sweep run を同時実行

# sweep が実行するトレーニング/チューニングジョブ
job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
method: grid
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
```
{{% /tab %}}
{{% tab "Custom scheduler" %}}
  カスタムスケジューラは scheduler-job を作成して構築します。このガイドでは `WandbScheduler` を修正してロギング機能を拡張する例を示します。

  1. `wandb/launch-jobs` リポジトリ（特に `wandb/launch-jobs/jobs/sweep_schedulers`）をクローンします
  2. `wandb_scheduler.py` を修正し、ロギングを追加するなど望む処理を実装します。例: 関数 `_poll` に追加ロギングを入れる（この関数はポーリングごとに呼び出され、新規 sweep run を launch する前に処理されます）。
  3. 修正後のファイルを `python wandb_scheduler.py --project <project> --entity <entity> --name CustomWandbScheduler` でジョブ作成として実行します
  4. 作成されたジョブの名前を、UI や先の実行出力などから調べます（特に指定しなければ code-artifact job になります）
  5. スケジューラがこの新しいジョブを指すように sweep 設定を作成します

```yaml
...
scheduler:
  job: '<entity>/<project>/job-CustomWandbScheduler:latest'
...
```

{{% /tab %}}
{{% tab "Optuna scheduler" %}}

  Optuna は様々なアルゴリズムでモデルに最良のハイパーパラメータを探索するハイパーパラメータ最適化フレームワークです（W&B と似ています）。[サンプラーアルゴリズム](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) に加えて、[プルーニングアルゴリズム](https://optuna.readthedocs.io/en/stable/reference/pruners.html) も組み込みで提供され、高速化のためにパフォーマンスの悪い run を早期停止できます。多くの run を実行する場合に、時間やリソースを大幅に節約できます。クラスは細かく設定でき、`scheduler.settings.pruner/sampler.args` ブロックに必要なパラメータを指定します。



Optuna のスケジューリングロジックを利用して launch sweep を作成する手順です。

1. まず独自のジョブを作成するか、Optuna スケジューラのイメージジョブ（プリビルド）を使用します。
    * 独自ジョブの作成方法例は、[`wandb/launch-jobs`](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers) リポジトリをご覧ください。
    * プリビルドの Optuna イメージを使う場合、`wandb/sweep-jobs` プロジェクトの `job-optuna-sweep-scheduler` か、ジョブ名 `wandb/sweep-jobs/job-optuna-sweep-scheduler:latest` をそのままご利用いただけます。
    

2. ジョブ作成後、sweep 設定ファイルを作成します。`scheduler` ブロックを追加し、`job` キーで Optuna スケジューラジョブの名前を指定します（例は以下）。

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
    resource: local-container  # イメージをソースにする scheduler ジョブには必須
    num_workers: 2

    # optuna 固有の設定
    settings:
      pruner:
        type: PercentilePruner
        args:
          percentile: 25.0  # run の上位25%を残して他は停止
          n_warmup_steps: 10  # 最初の x ステップはプルーニングしない

  parameters:
    learning_rate:
      min: 0.0001
      max: 0.1
  ```


  3. 最後に、`launch-sweep` コマンドで sweep をアクティブなキューに送信します:
  
  ```bash
  wandb launch-sweep <config.yaml> -q <queue> -p <project> -e <entity>
  ```

  Optuna sweep scheduler ジョブの実装は [wandb/launch-jobs](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers/optuna_scheduler/optuna_scheduler.py) をご覧ください。Optuna スケジューラの利用例やカスタマイズ例は [wandb/examples](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/optuna-scheduler) でも紹介しています。
{{% /tab %}}
{{< /tabpane >}}

カスタム sweep スケジューラジョブの実装例は [wandb/launch-jobs](https://github.com/wandb/launch-jobs) リポジトリの `jobs/sweep_schedulers` フォルダで確認できます。このガイドでは公開されている **Wandb Scheduler Job** の利用方法だけでなく、独自の sweep スケジューラジョブの作成プロセスも紹介しています。


## launch 上で sweep を再開する方法
既に launch された sweep から launch-sweep を再開することもできます。ハイパーパラメータやトレーニングジョブは変更できませんが、スケジューラ専用パラメータや投入先のキューは変更可能です。

{{% alert %}}
初回の sweep で 'latest' のようなエイリアス付きジョブを使っていた場合、その後のジョブバージョン更新で再開時の実行結果が異なることがあります。
{{% /alert %}}

1. 以前実行した launch sweep の sweep 名/ID を特定します。sweep ID は8文字の英数字（例 `hhd16935`）で、W&B App のプロジェクトで確認できます。
2. スケジューラパラメータを変更したい場合、更新済みの config ファイルを用意します。
3. ターミナルで下記コマンドを実行します。`<` と `>` はご自身の情報で置き換えてください。

```bash
wandb launch-sweep <optional config.yaml> --resume_id <sweep id> --queue <queue_name>
```