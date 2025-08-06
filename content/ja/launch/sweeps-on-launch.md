---
title: W&B Launch でスイープを作成する
description: ローンチでハイパーパラメータのスイープを自動化する方法をご紹介します。
menu:
  launch:
    identifier: ja-launch-sweeps-on-launch
    parent: launch
url: guides/launch/sweeps-on-launch
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1WxLKaJlltThgZyhc7dcZhDQ6cjVQDfil#scrollTo=AFEzIxA6foC7" >}}

W&B Launch でハイパーパラメータチューニングジョブ（[sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}})）を作成しましょう。Launch 上の sweep では、指定したハイパーパラメータで sweep する sweep scheduler が Launch Queue に投入されます。エージェントが sweep scheduler をピックアップすると、その選択されたハイパーパラメータで sweep run を同じ queue に投入します。sweep が完了するか停止されるまで、このプロセスは繰り返されます。

デフォルトの W&B Sweep スケジューリングエンジンを使用することも、自分でカスタムスケジューラを実装することも可能です。

1. 標準 sweep scheduler: デフォルトの W&B Sweep スケジューリングエンジンを使用します。[W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}})でお馴染みの `bayes`、`grid`、`random` メソッドが利用できます。
2. カスタム sweep scheduler: sweep scheduler をジョブとして実行できるように設定します。このオプションでは完全にカスタマイズ可能です。標準の sweep scheduler を拡張してログを追加する方法については、下記のセクションもご参照ください。

{{% alert %}}
このガイドでは、事前に W&B Launch の設定が完了していることを前提としています。W&B Launch の設定がまだの場合は、launch ドキュメントの[開始方法]({{< relref path="./#how-to-get-started" lang="ja" >}})セクションをご覧ください。
{{% /alert %}}

{{% alert %}}
初めて Launch で sweep を作成する場合は「basic」メソッドでの sweep 作成をおすすめします。標準の W&B scheduling エンジンで対応できない場合のみ、カスタムの sweeps on launch scheduler をご利用ください。
{{% /alert %}}

## W&B 標準スケジューラーで sweep を作成する
Launch で W&B Sweeps を作成します。W&B App のインタラクティブ UI もしくは W&B CLI からプログラム的にも sweep を作成できます。scheduler のカスタマイズ等、高度な設定が必要な場合は CLI をご利用ください。

{{% alert %}}
W&B Launch で sweep を作成する前に、まず sweep 対象となるジョブを先に作成しておく必要があります。詳細は[ジョブ作成ページ]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ja" >}})をご覧ください。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B app" %}}

W&B App を使ってインタラクティブに sweep を作成します。

1. W&B App で自分のプロジェクトにアクセスします。  
2. 左パネル（ほうきアイコン）から sweeps アイコンを選択します。
3. **Create Sweep** ボタンをクリックします。
4. **Configure Launch** ボタンをクリックします。
5. **Job** ドロップダウンから sweep 対象の job 名とそのバージョンを選択します。
6. **Queue** ドロップダウンで sweep を実行する queue を選択します。
7. **Job Priority** ドロップダウンで launch job の優先度を設定します。launch queue で優先度指定がない場合は"Medium"がデフォルトです。
8. （任意）run や sweep scheduler の override 引数を設定できます。例えば scheduler の override で `num_workers` によって同時に管理する run 数を調整できます。
9. （任意）**Destination Project** ドロップダウンで sweep の保存先プロジェクトを選択します。
10. **Save** をクリックします。
11. **Launch Sweep** を選択します。

{{< img src="/images/launch/create_sweep_with_launch.png" alt="Launch sweep configuration" >}}

{{% /tab %}}
{{% tab "CLI" %}}

W&B CLI で W&B Sweep を Launch 上でプログラム的に作成します。

1. Sweep の設定ファイルを作成します
2. sweep 設定ファイル内に Job 名を記載します
3. sweep agent を初期化します

{{% alert %}}
ステップ1,3は通常の W&B Sweep 作成時と同様の流れです。
{{% /alert %}}

以下のコードスニペットのように、job に `'wandb/jobs/Hello World 2:latest'` を指定します。

```yaml
# launch-sweep-config.yaml

job: 'wandb/jobs/Hello World 2:latest'
description: launch jobs を使った sweep のサンプル

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

# スケジューラーのオプションパラメータ例:

# scheduler:
#   num_workers: 1  # 同時 sweep run 数
#   docker_image: <スケジューラー用base image>
#   resource: <例: local-container...>
#   resource_args:  # run に渡されるリソース引数
#     env: 
#         - WANDB_API_KEY

# Launch のオプションパラメータ例
# launch: 
#    registry: <image取得用レジストリ>
```

sweep configuration の作成方法については[Define sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration.md" lang="ja" >}})ページも参照ください。

4. 次に sweep を初期化します。config ファイルパス、ジョブキュー名、W&B Entity 名、Project 名を指定して実行します。

```bash
wandb launch-sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>
```

W&B Sweeps の詳細は[ハイパーパラメータのチューニング]({{< relref path="/guides/models/sweeps/" lang="ja" >}})チャプターもご覧ください。

{{% /tab %}}
{{< /tabpane >}}


## カスタム sweep スケジューラーの作成
W&B scheduler 又は独自 scheduler を使ってカスタム sweep scheduler を作成できます。

{{% alert %}}
scheduler ジョブの利用には wandb cli バージョン `0.15.4` 以上が必要です
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B scheduler" %}}
  W&B の sweep scheduling ロジックをジョブとして利用し、launch sweep を作成します。
  
  1. public の wandb/sweep-jobs プロジェクトの Wandb scheduler job を特定するか、以下のジョブ名を利用します:
  `'wandb/sweep-jobs/job-wandb-sweep-scheduler:latest'`
  2. このジョブ名を `scheduler` ブロックの `job` キーに含めた configuration yaml を作成します（下記例を参照）。
  3. 作成した config を指定し `wandb launch-sweep` コマンドを実行します。

設定例:
```yaml
# launch-sweep-config.yaml  
description: スケジューラージョブを使った launch sweep 設定例
scheduler:
  job: wandb/sweep-jobs/job-wandb-sweep-scheduler:latest
  num_workers: 8  # 最大8個の sweep run 同時実行

# sweep run が実行するトレーニング/チューニング用ジョブ
job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
method: grid
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
```
{{% /tab %}}
{{% tab "Custom scheduler" %}}
  scheduler-job を作成してカスタム scheduler を作ることができます。このガイドでは `WandbScheduler` を修正し、より多くのログを出力する例を紹介します。

  1. `wandb/launch-jobs` リポジトリ（`wandb/launch-jobs/jobs/sweep_schedulers`）をクローンします。
  2. `wandb_scheduler.py` を修正し、ログ出力を増やすことができます。例：関数 `_poll` にロギングを追加。これは（設定可能な間隔で）各ポーリングサイクルの前、sweep run を launch する直前に呼び出されます。
  3. 修正したファイルを使ってジョブを作成します。 `python wandb_scheduler.py --project <project> --entity <entity> --name CustomWandbScheduler`
  4. 作成されたジョブ名（UI またはコマンド出力に表示される、code-artifact ジョブの場合が多い）を特定します。
  5. scheduler が新しいジョブを指すよう sweep 設定に反映します。

```yaml
...
scheduler:
  job: '<entity>/<project>/job-CustomWandbScheduler:latest'
...
```

{{% /tab %}}
{{% tab "Optuna scheduler" %}}

  Optuna は様々なアルゴリズムで最適なハイパーパラメータを探すハイパーパラメータ最適化フレームワークです（W&Bと似ています）。[サンプリングアルゴリズム](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html)に加え、[プルーニングアルゴリズム](https://optuna.readthedocs.io/en/stable/reference/pruners.html)も活用でき、成績が悪い run を早期に終了させることで試行回数が多いケースでも効率的にリソース・時間を節約できます。各クラスは柔軟に設定でき、`scheduler.settings.pruner/sampler.args` ブロックで必要なパラメータを渡せます。

Optuna の scheduling ロジックを使って launch sweep を実行します。

1. まず自分でジョブを作成するか、Optuna scheduler イメージの既成ジョブを利用できます。  
    * オリジナルジョブを作りたい場合は [`wandb/launch-jobs`](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers) レポジトリの例も参照ください。
    * 既成の Optuna イメージを使う場合は、`wandb/sweep-jobs` プロジェクトの `job-optuna-sweep-scheduler` にアクセスするか、ジョブ名 `wandb/sweep-jobs/job-optuna-sweep-scheduler:latest` を利用できます。
    

2. ジョブを作成したら sweep 設定を行います。`scheduler` ブロックで `job` キーが Optuna scheduler ジョブを指す形に設定します（例↓）。

```yaml
  # optuna_config_basic.yaml
  description: シンプルな Optuna scheduler
  job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
  run_cap: 5
  metric:
    name: epoch/val_loss
    goal: minimize

  scheduler:
    job: wandb/sweep-jobs/job-optuna-sweep-scheduler:latest
    resource: local-container  # イメージ由来の scheduler job 用
    num_workers: 2

    # optuna 固有の設定
    settings:
      pruner:
        type: PercentilePruner
        args:
          percentile: 25.0  # 上位25%のみ残す
          n_warmup_steps: 10  # 最初の x ステップは pruning 無効

  parameters:
    learning_rate:
      min: 0.0001
      max: 0.1
  ```


  3. 最後に、launch-sweep コマンドを使って sweep をアクティブな queue で実行します。
  
  ```bash
  wandb launch-sweep <config.yaml> -q <queue> -p <project> -e <entity>
  ```


  Optuna sweep scheduler ジョブの実装詳細は [wandb/launch-jobs](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers/optuna_scheduler/optuna_scheduler.py)をご覧ください。Optuna scheduler でできることの例は [wandb/examples](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/optuna-scheduler) でも紹介しています。
{{% /tab %}}
{{< /tabpane >}}

 カスタム sweep scheduler job でできることの例は [wandb/launch-jobs](https://github.com/wandb/launch-jobs) レポジトリの `jobs/sweep_schedulers` 下にあります。このガイドでは公開中の **Wandb Scheduler Job** の使い方、独自ジョブの作成手順も紹介しています。

 ## sweep を Launch で再開する方法
  以前に実行した launch-sweep から sweep を再開することもできます。ハイパーパラメータやトレーニングジョブは変更できませんが、スケジューラー固有のパラメータや、push 先 queue の変更は可能です。

{{% alert %}}
初回 sweep が 'latest' のようなエイリアス付き training job を利用していた場合、最新バージョンが変わった状態で再開すると過去と異なる結果となる可能性があります。
{{% /alert %}}

  1. 以前に実施した launch sweep の sweep 名（ID）を特定します。sweep ID は8文字の文字列（例: `hhd16935`）で、W&B App 上の該当プロジェクトでも確認できます。
  2. スケジューラーのパラメータに変更がある場合は、更新した config ファイルを作成します。
  3. ターミナルで以下のコマンドを実行してください（`< >` で示した箇所は自分の情報に置き換えてください）:

```bash
wandb launch-sweep <optional config.yaml> --resume_id <sweep id> --queue <queue_name>
```