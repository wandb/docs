---
title: W&B Launch を使って Sweeps を作成する
description: ハイパーパラメータの Sweeps を Launch で自動化する方法を学びましょう。
menu:
  launch:
    identifier: ja-launch-sweeps-on-launch
    parent: launch
url: guides/launch/sweeps-on-launch
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1WxLKaJlltThgZyhc7dcZhDQ6cjVQDfil#scrollTo=AFEzIxA6foC7" >}}

W&B Launch でハイパーパラメータ チューニングの job（[Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}})）を作成します。Launch 上の sweeps では、指定したハイパーパラメータで探索する sweep スケジューラが Launch Queue に投入されます。エージェントにピックアップされるとスケジューラが起動し、選択されたハイパーパラメータで同じ Queue に sweep の run を起動します。これは sweep が完了するか停止されるまで続きます。

既定の W&B Sweep スケジューリング エンジンを使うか、独自のカスタム スケジューラを実装できます。

1. 標準 sweep スケジューラ: [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) を制御する既定の W&B Sweep スケジューリング エンジンを使用します。おなじみの `bayes`、`grid`、`random` メソッドが利用可能です。
2. カスタム sweep スケジューラ: スケジューラを job として実行するように設定します。このオプションにより完全なカスタマイズが可能です。標準の sweep スケジューラを拡張してログ出力を増やす例は以下のセクションにあります。
 
{{% alert %}}
このガイドは、W&B Launch が事前に設定済みであることを前提としています。W&B Launch の設定がまだの場合は、Launch ドキュメントの [開始方法]({{< relref path="./#how-to-get-started" lang="ja" >}}) を参照してください。
{{% /alert %}}

{{% alert %}}
Launch 上の sweeps を初めて使う場合は、まずは 'basic' メソッドで sweep を作成することをおすすめします。標準の W&B スケジューリング エンジンで要件を満たせない場合に、カスタムの Launch 上の sweeps スケジューラを使用してください。
{{% /alert %}}

## W&B 標準スケジューラで sweep を作成する
Launch で W&B Sweeps を作成します。W&B App から対話的に作成する方法と、W&B CLI を使ってプログラムから作成する方法があります。スケジューラのカスタマイズなど Launch の sweep を高度に設定するには、CLI を使用してください。

{{% alert %}}
W&B Launch で sweep を作成する前に、まず sweep の対象となる job を作成してください。詳しくは [Create a Job]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B App" %}}

W&B App から対話的に sweep を作成します。

1. W&B App で対象の W&B Project に移動します。  
2. 左パネルの Sweeps アイコン（ほうきの画像）を選択します。
3. 次に、**Create Sweep** ボタンを選択します。
4. **Configure Launch** ボタンをクリックします。
5. **Job** ドロップダウンから、sweep の元にする job 名とそのバージョンを選択します。
6. **Queue** ドロップダウンから、sweep を実行する Queue を選択します。
8. **Job Priority** ドロップダウンで Launch job の優先度を指定します。Launch Queue が優先度に対応していない場合、Launch job の優先度は "Medium" に設定されます。
8. （任意）run または sweep スケジューラに対する override 引数を設定します。例えば、scheduler の override を使って、スケジューラが管理する同時実行 run 数を `num_workers` で設定できます。
9. （任意）**Destination Project** ドロップダウンから、sweep を保存する Project を選択します。
10. **Save** をクリックします。
11. **Launch Sweep** を選択します。

{{< img src="/images/launch/create_sweep_with_launch.png" alt="Launch の sweep 設定" >}}

{{% /tab %}}
{{% tab "CLI" %}}

W&B CLI を使って、Launch で W&B Sweep をプログラムから作成します。

1. Sweep configuration を作成する
2. sweep configuration 内で job のフルネームを指定する
3. sweep エージェントを初期化する

{{% alert %}}
1 と 3 は、通常の W&B Sweep 作成時と同じ手順です。
{{% /alert %}}

例えば、次のコードスニペットでは job の値に `'wandb/jobs/Hello World 2:latest'` を指定しています。

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

# オプションのスケジューラ パラメータ:

# scheduler:
#   num_workers: 1  # 同時実行する sweep run 数
#   docker_image: <スケジューラ用のベースイメージ>
#   resource: <例: local-container など>
#   resource_args:  # run に渡すリソース引数
#     env: 
#         - WANDB_API_KEY

# オプションの Launch パラメータ
# launch: 
#    registry: <イメージ取得に使うレジストリ>
```

sweep configuration の作成方法は、[Define sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration.md" lang="ja" >}}) を参照してください。

4. 次に、sweep を初期化します。設定ファイルへのパス、job Queue 名、W&B Entity、Project 名を指定します。

```bash
wandb launch-sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>
```

W&B Sweeps の詳細は、[ハイパーパラメータをチューニングする]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) チャプターを参照してください。

{{% /tab %}}
{{< /tabpane >}}


## カスタム sweep スケジューラを作成する
W&B のスケジューラ job または独自のカスタム スケジューラで、カスタム sweep スケジューラを作成できます。

{{% alert %}}
スケジューラ job の利用には wandb CLI バージョンが `0.15.4` 以上である必要があります。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B スケジューラ" %}}
  W&B の sweep スケジューリング ロジックを job として使い、Launch の sweep を作成します。
  
  1. 公開プロジェクト wandb/sweep-jobs の中から Wandb スケジューラ job を探すか、次の job 名を使用します:
  `'wandb/sweep-jobs/job-wandb-sweep-scheduler:latest'`
  2. この名前を指す `job` キーを含む `scheduler` ブロックを追加した configuration YAML を作成します（下記例）。
  3. 新しい設定で `wandb launch-sweep` コマンドを使用します。


例の設定:
```yaml
# launch-sweep-config.yaml  
description: Launch sweep config using a scheduler job
scheduler:
  job: wandb/sweep-jobs/job-wandb-sweep-scheduler:latest
  num_workers: 8  # 8 個の sweep run を同時実行

# sweep の run が実行するトレーニング/チューニング用の job
job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
method: grid
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
```
{{% /tab %}}
{{% tab "カスタム スケジューラ" %}}
  カスタム スケジューラは scheduler-job を作成することで作れます。本ガイドでは、`WandbScheduler` を変更してログ出力を増やす例を扱います。 

  1. `wandb/launch-jobs` リポジトリをクローンします（特に `wandb/launch-jobs/jobs/sweep_schedulers`）
  2. 目的の追加ログ出力を得るために `wandb_scheduler.py` を変更します。例: 関数 `_poll` にログを追加します。これは新しい sweep run を起動する前、各ポーリング サイクル（間隔は設定可）で 1 回呼び出されます。
  3. 変更したファイルを実行して job を作成します: `python wandb_scheduler.py --project <project> --entity <entity> --name CustomWandbScheduler`
  4. 作成された job の名前を、UI か前のコマンド出力で確認します。特に指定しない限り code-artifact Job になります。
  5. `scheduler` が新しい job を指すように、sweep configuration を作成します。

```yaml
...
scheduler:
  job: '<entity>/<project>/job-CustomWandbScheduler:latest'
...
```

{{% /tab %}}
{{% tab "Optuna スケジューラ" %}}

  Optuna は、与えられた model に対して最適なハイパーパラメータを見つけるために様々なアルゴリズムを用いるハイパーパラメータ最適化フレームワークです（W&B と同様）。[サンプリング アルゴリズム](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) に加えて、Optuna には成績の悪い run を早期に終了できる様々な [プルーニング アルゴリズム](https://optuna.readthedocs.io/en/stable/reference/pruners.html) も用意されています。大量の run を実行する際に、時間とリソースの節約に特に有効です。これらのクラスは高い柔軟性を持ち、設定ファイルの `scheduler.settings.pruner/sampler.args` ブロックに想定パラメータを渡すだけで構いません。



Optuna のスケジューリング ロジックを job で使って、Launch の sweep を作成します。

1. まず、自分用の job を作成するか、あらかじめ用意された Optuna スケジューラのイメージ job を使用します。 
    * 独自の job の作り方は、[`wandb/launch-jobs`](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers) リポジトリの例を参照してください。
    * 事前ビルド済みの Optuna イメージを使うには、`wandb/sweep-jobs` プロジェクト内の `job-optuna-sweep-scheduler` に移動するか、job 名 `wandb/sweep-jobs/job-optuna-sweep-scheduler:latest` を使用します。 
    

2. job を作成したら、次に sweep を作成します。`scheduler` ブロックを含み、`job` キーが Optuna スケジューラの job を指す sweep の config を作成します（以下の例）。

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
    resource: local-container  # イメージから取得するスケジューラ job には必須
    num_workers: 2

    # Optuna 固有の設定
    settings:
      pruner:
        type: PercentilePruner
        args:
          percentile: 25.0  # run の 75% を打ち切る
          n_warmup_steps: 10  # 最初の x ステップはプルーニングを無効化

  parameters:
    learning_rate:
      min: 0.0001
      max: 0.1
  ```

  3. 最後に、`launch-sweep` コマンドでアクティブな Queue に sweep を投入します。
  
  ```bash
  wandb launch-sweep <config.yaml> -q <queue> -p <project> -e <entity>
  ```

  Optuna sweep スケジューラ job の実装は [wandb/launch-jobs](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers/optuna_scheduler/optuna_scheduler.py) を参照してください。Optuna スケジューラで可能なことの例は [wandb/examples](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/optuna-scheduler) もご覧ください。
{{% /tab %}}
{{< /tabpane >}}

カスタム sweep スケジューラ job の例は、`jobs/sweep_schedulers` 配下の [wandb/launch-jobs](https://github.com/wandb/launch-jobs) リポジトリで公開されています。本ガイドでは、公開されている「Wandb Scheduler Job」の使い方と、カスタム sweep スケジューラ job を作成する手順の一例を紹介します。

## Launch で sweep を再開する方法
過去に起動した sweep から、launch-sweep を再開することもできます。ハイパーパラメータやトレーニング job は変更できませんが、スケジューラ固有のパラメータや、投入先の Queue は変更できます。

{{% alert %}}
最初の sweep で 'latest' のようなエイリアス付きのトレーニング job を使用していた場合、前回実行以降に最新の job バージョンが更新されていると、再開後の結果が異なる可能性があります。
{{% /alert %}}

1. 過去に実行した launch sweep の名前/ID を確認します。sweep ID は 8 文字の文字列（例: `hhd16935`）で、W&B App の Project 内で確認できます。
2. スケジューラのパラメータを変更する場合は、更新した config ファイルを作成します。
3. ターミナルで次のコマンドを実行します。`<` と `>` で囲まれた部分は自身の情報に置き換えてください。

```bash
wandb launch-sweep <optional config.yaml> --resume_id <sweep id> --queue <queue_name>
```