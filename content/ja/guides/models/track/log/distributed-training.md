---
title: 分散トレーニングの実験をログに記録する
description: W&B を使って、複数の GPU による分散トレーニング実験をログします。
menu:
  default:
    identifier: ja-guides-models-track-log-distributed-training
    parent: log-objects-and-media
---

分散トレーニングの実験では、複数のマシンやクライアントを並列に使って モデル をトレーニングします。W&B は分散トレーニングの 実験 をトラッキングするのに役立ちます。ユースケースに応じて、次のいずれかの方法で分散トレーニングの 実験 をトラッキングしてください。

* **単一プロセスをトラッキング**: W&B で rank 0 プロセス（"leader" や "coordinator" とも呼ばれる）をトラッキングします。これは [PyTorch Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel)（DDP）クラスを使った分散トレーニングのログにおける一般的な解決策です。
* **複数プロセスをトラッキング**: 複数プロセスの場合は、次のいずれかを選べます:
   * 各プロセスごとに 1 つの run を使って個別にトラッキングします。W&B App UI で任意にまとめてグループ化できます。
   * すべてのプロセスを 1 つの run にトラッキングします。

## 単一プロセスをトラッキング

このセクションでは、rank 0 プロセスで利用できる 値 や メトリクス のトラッキング方法を説明します。単一プロセスから利用できるメトリクスのみをトラッキングしたい場合にこの方法を使ってください。典型的なメトリクスには GPU/CPU の使用率、共有 検証セット 上での振る舞い、勾配とパラメータ、代表的な データ サンプルでの損失値などがあります。

rank 0 プロセス内で、[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) を使って W&B の run を初期化し、その run に対して [`wandb.log`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog" lang="ja" >}}) でログを記録します。

[サンプルの Python スクリプト（`log-ddp.py`）](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) は、単一マシン上の 2 枚の GPU で PyTorch DDP を使ってメトリクスをトラッキングする方法の一例を示しています。[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)（`torch.nn` の `DistributedDataParallel`）は、分散トレーニングのための一般的な ライブラリ です。基本的な考え方はどの分散トレーニングのセットアップにも当てはまりますが、実装は異なる場合があります。

この Python スクリプトは次のことを行います:
1. `torch.distributed.launch` で複数プロセスを起動します。
1. `--local_rank` コマンドライン 引数で rank を確認します。
1. rank が 0 の場合、[`train()`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py#L24) 関数内で条件付きに `wandb` のログを設定します。

```python
if __name__ == "__main__":
    # 引数を取得
    args = parse_args()

    if args.local_rank == 0:  # メインプロセスのみ
        # wandb の run を初期化
        run = wandb.init(
            entity=args.entity,
            project=args.project,
        )
        # DDP でモデルをトレーニング
        train(args, run)
    else:
        train(args)
```

[単一プロセスからトラッキングされたメトリクスを表示する例のダッシュボード](https://wandb.ai/ayush-thakur/DDP/runs/1s56u3hc/system)を参照してください。

このダッシュボードには、温度や使用率など、両方の GPU のシステムメトリクスが表示されます。

{{< img src="/images/track/distributed_training_method1.png" alt="GPU メトリクスのダッシュボード" >}}

ただし、エポック と バッチサイズ の関数としての損失値は単一の GPU からのみログされました。

{{< img src="/images/experiments/loss_function_single_gpu.png" alt="損失関数のプロット" >}}

## 複数プロセスをトラッキング

W&B で複数プロセスをトラッキングするには、次のいずれかの方法を使います:
* 各プロセスごとに run を作成して[個別にトラッキングする]({{< relref path="distributed-training/#track-each-process-separately" lang="ja" >}})
* [すべてのプロセスを 1 つの run にトラッキングする]({{< relref path="distributed-training/#track-all-processes-to-a-single-run" lang="ja" >}})

### 各プロセスを個別にトラッキング

このセクションでは、各プロセスに対して 1 つの run を作成し、個別にトラッキングする方法を説明します。各 run では、その run に対してメトリクスや Artifacts などをログします。トレーニングの最後に `wandb.Run.finish()` を呼び出して、run の完了をマークし、すべてのプロセスが正しく終了できるようにします。

複数の 実験 を横断して run を追いかけるのが難しいことがあります。その場合は、W&B を初期化する際に `group` パラメータに 値 を設定（`wandb.init(group='group-name')`）して、どの run がどの 実験 に属するかを識別できるようにしてください。W&B の 実験 におけるトレーニングと評価の Runs を整理する方法の詳細は、[Group Runs]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) を参照してください。

{{% alert %}}
**個々のプロセスからのメトリクスをトラッキングしたい場合は、この方法を使ってください。** 典型的な例として、（データ分配のデバッグのための）各ノードの データ と 予測、メインノードの外での個々のバッチのメトリクスなどがあります。すべてのノードのシステムメトリクスを取得したり、メインノードで利用できる要約統計量を取得したりするために、この方法は必須ではありません。
{{% /alert %}}

次の Python のコードスニペットは、W&B を初期化するときに group パラメータを設定する方法を示しています。

```python
if __name__ == "__main__":
    # 引数を取得
    args = parse_args()
    # run を初期化
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group="DDP",  # この実験のすべての run を 1 つのグループにまとめる
    )
    # DDP でモデルをトレーニング
    train(args, run)

    run.finish()  # run を完了としてマーク
```

W&B App UI で、複数プロセスからトラッキングされたメトリクスの[例のダッシュボード](https://wandb.ai/ayush-thakur/DDP?workspace=user-noahluna)を確認してください。左サイドバーには 2 つの W&B Runs が 1 つにグループ化されていることが分かります。グループをクリックすると、その 実験 の専用グループページを表示できます。専用グループページでは、各プロセスのメトリクスが個別に表示されます。

{{< img src="/images/experiments/dashboard_grouped_runs.png" alt="分散 run のグループ表示" >}}

上の画像は W&B App UI のダッシュボードを示しています。サイドバーには 2 つの 実験 が表示され、1 つは 'null'、もう 1 つ（黄色の枠で囲まれている）は 'DPP' とラベル付けされています。グループを展開（Group ドロップダウンを選択）すると、その 実験 に紐づく W&B Runs が表示されます。

### すべてのプロセスを 1 つの run にトラッキング

{{% alert color="secondary"  %}}
`x_` で始まるパラメータ（`x_label` など）はパブリックプレビューです。フィードバックは [W&B リポジトリの GitHub issue](https://github.com/wandb/wandb) に作成してください。
{{% /alert %}}

{{% alert title="要件" %}}
複数プロセスを 1 つの run にトラッキングするには、以下が必要です:
- W&B Python SDK バージョン `v0.19.9` 以降

- W&B Server v0.68 以降
{{% /alert  %}}

この方法では、プライマリノードと 1 台以上のワーカーノードを使います。プライマリノードで W&B の run を初期化します。各ワーカーノードでは、プライマリノードで使用した run ID を使って run を初期化します。トレーニング中、各ワーカーノードはプライマリノードと同じ run ID にログを書き込みます。W&B はすべてのノードからのメトリクスを集約し、W&B App UI に表示します。

プライマリノードでは、[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) で W&B の run を初期化します。`settings` パラメータに `wandb.Settings` オブジェクト（`wandb.init(settings=wandb.Settings()`）を渡し、以下を設定します。

1. 共有モードを有効にするため、`mode` パラメータを `"shared"` に設定します。
2. [`x_label`](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py#L638) にユニークなラベルを設定します。W&B App UI のログやシステムメトリクスで、どのノードからのデータかを識別するために `x_label` の 値 が使われます。未指定の場合、W&B はホスト名とランダムなハッシュを用いてラベルを自動作成します。
3. このノードがプライマリであることを示すため、[`x_primary`](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py#L660) を `True` に設定します。
4. 任意で、W&B がメトリクスをトラッキングする GPU を指定するために `x_stats_gpu_device_ids` に GPU インデックスのリスト（例: [0,1,2]）を設定します。リストを指定しない場合、W&B はマシン上のすべての GPU をトラッキングします。

プライマリノードの run ID を控えておいてください。各ワーカーノードにはプライマリノードの run ID が必要です。

{{% alert %}}
`x_primary=True` はプライマリノードとワーカーノードを区別します。プライマリノードは、設定ファイルやテレメトリーなど、ノード間で共有されるファイルのアップロードを行う唯一のノードです。ワーカーノードはこれらのファイルをアップロードしません。
{{% /alert %}}

各ワーカーノードでは、[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) で W&B の run を初期化し、以下を指定します。
1. `settings` パラメータに `wandb.Settings` オブジェクト（`wandb.init(settings=wandb.Settings()`）を渡し、以下を設定:
   * 共有モードを有効にするため、`mode` パラメータを `"shared"` に設定。
   * `x_label` にユニークなラベルを設定。W&B App UI のログやシステムメトリクスで、どのノードからのデータかを識別するために `x_label` の 値 が使われます。未指定の場合、W&B はホスト名とランダムなハッシュを用いてラベルを自動作成します。
   * このノードがワーカーであることを示すため、`x_primary` を `False` に設定。
2. `id` パラメータに、プライマリノードで使用した run ID を渡します。
3. 任意で [`x_update_finish_state`](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py#L772) を `False` に設定します。これにより、非プライマリノードが run の[状態]({{< relref path="/guides/models/track/runs/#run-states" lang="ja" >}})を早まって `finished` に更新してしまうことを防ぎ、run の状態をプライマリノードで一貫して管理できるようにします。

{{% alert %}}
プライマリノードの run ID を設定するために 環境変数 を使用し、各ワーカーノードのマシンでそれを定義する方法を検討してください。
{{% /alert %}}

以下のサンプルコードは、複数プロセスを 1 つの run にトラッキングするための高レベルの要件を示しています。

```python
import wandb

# プライマリノードで run を初期化
run = wandb.init(
    entity="entity",
    project="project",
	settings=wandb.Settings(
        x_label="rank_0", 
        mode="shared", 
        x_primary=True,
        x_stats_gpu_device_ids=[0, 1],  # （任意）GPU 0 と 1 のメトリクスのみをトラッキング
        )
)

# プライマリノードの run ID を控える
# 各ワーカーノードにはこの run ID が必要
run_id = run.id

# プライマリノードの run ID を使ってワーカーノードで run を初期化
run = wandb.init(
	settings=wandb.Settings(x_label="rank_1", mode="shared", x_primary=False),
	id=run_id,
)

# プライマリノードの run ID を使ってワーカーノードで run を初期化
run = wandb.init(
	settings=wandb.Settings(x_label="rank_2", mode="shared", x_primary=False),
	id=run_id,
)
```

実際のユースケースでは、各ワーカーノードは別々のマシン上にあることが多いでしょう。

{{% alert %}}
GKE 上の マルチノード・マルチ GPU の Kubernetes クラスターで モデル をトレーニングするエンドツーエンドの例については、[Distributed Training with Shared Mode](https://wandb.ai/dimaduev/simple-cnn-ddp/reports/Distributed-Training-with-Shared-Mode--VmlldzoxMTI0NTE1NA) のレポートを参照してください。
{{% /alert %}}

run がログを書き込んでいる Project 内で、マルチノードプロセスのコンソールログを表示するには:

1. 該当の run を含む Project に移動します。
2. 左サイドバーで **Runs** タブをクリックします。
3. 表示したい run をクリックします。
4. 左サイドバーで **Logs** タブをクリックします。

コンソールログは、UI のコンソールログページ上部にある検索バーで、`x_label` に設定したラベルに基づいてフィルタできます。たとえば、`x_label` に `rank0`、`rank1`、`rank2`、`rank3`、`rank4`、`rank5`、`rank6` を与えた場合、次の画像のようなフィルタオプションが利用できます。`

{{< img src="/images/track/multi_node_console_logs.png" alt="マルチノードのコンソールログ" >}}

詳細は [Console logs]({{< relref path="/guides/models/app/console-logs/" lang="ja" >}}) を参照してください。

W&B はすべてのノードからのシステムメトリクスを集約し、W&B App UI に表示します。たとえば、次の画像は複数ノードのシステムメトリクスを持つサンプルダッシュボードを示しています。各ノードには `x_label` パラメータで指定した固有のラベル（`rank_0`、`rank_1`、`rank_2`）が付与されています。

{{< img src="/images/track/multi_node_system_metrics.png" alt="マルチノードのシステムメトリクス" >}}

折れ線グラフ パネルのカスタマイズ方法は [Line plots]({{< relref path="/guides/models/app/features/panels/line-plot/" lang="ja" >}}) を参照してください。

## ユースケース例

以下のコードスニペットは、高度な分散ユースケースでよくあるシナリオを示します。

### プロセスを生成

spawn されたプロセス内で run を初期化する場合は、メイン関数で `wandb.setup()` メソッドを使います。

```python
import multiprocessing as mp

def do_work(n):
    with wandb.init(config=dict(n=n)) as run:
        run.log(dict(this=n * n))

def main():
    wandb.setup()
    pool = mp.Pool(processes=4)
    pool.map(do_work, range(4))


if __name__ == "__main__":
    main()
```

### run を共有

プロセス間で run を共有するには、run オブジェクトを 引数 として渡します。

```python
def do_work(run):
    with wandb.init() as run:
        run.log(dict(this=1))

def main():
    run = wandb.init()
    p = mp.Process(target=do_work, kwargs=dict(run=run))
    p.start()
    p.join()
    run.finish()  # run を完了としてマーク


if __name__ == "__main__":
    main()
```

W&B はログの順序を保証しません。同期はスクリプトの作者が行ってください。

## トラブルシューティング

W&B と分散トレーニングを使用する際によくある問題は 2 つあります。

1. **トレーニング開始時にハングする** - 分散トレーニングのマルチプロセス処理と `wandb` のマルチプロセス処理が干渉すると、`wandb` のプロセスがハングすることがあります。
2. **トレーニング終了時にハングする** - `wandb` プロセスがいつ終了すべきか分からない場合、トレーニングジョブがハングすることがあります。Python スクリプトの最後で `wandb.Run.finish()` API を呼び出して、run が終了したことを W&B に知らせてください。`wandb.Run.finish()` API はデータのアップロードを完了し、W&B を終了させます。
分散ジョブの信頼性向上のために、W&B は `wandb service` コマンドの使用を推奨します。上記 2 つのトレーニングの問題は、wandb service が利用できない W&B SDK のバージョンでよく見られます。

### W&B Service を有効化

W&B SDK のバージョンによっては、W&B Service が既定で有効になっている場合があります。

#### W&B SDK 0.13.0 以上

W&B SDK バージョン `0.13.0` 以上では、W&B Service は既定で有効です。

#### W&B SDK 0.12.5 以上

W&B SDK バージョン 0.12.5 以上で W&B Service を有効にするには、Python スクリプトを変更します。`wandb.require` メソッドを使い、メイン関数内で文字列 `"service"` を渡してください。

```python
if __name__ == "__main__":
    main()


def main():
    wandb.require("service")
    # この下にスクリプト本体を記述
```

最適な体験のため、可能であれば最新バージョンへのアップグレードを推奨します。

**W&B SDK 0.12.4 以下**

W&B SDK バージョン 0.12.4 以下を使用している場合は、マルチスレッドを使うために `WANDB_START_METHOD` 環境変数を `"thread"` に設定してください。