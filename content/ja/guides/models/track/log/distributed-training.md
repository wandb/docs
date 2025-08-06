---
title: 分散トレーニング実験をログする
description: W&B を使って、複数 GPU を用いた分散トレーニング実験のログを記録しましょう。
menu:
  default:
    identifier: distributed-training
    parent: log-objects-and-media
---

分散トレーニング実験では、複数のマシンやクライアントを並列で使ってモデルをトレーニングします。W&B は分散トレーニング実験のトラッキングをサポートしています。お使いのユースケースに合わせて、以下のいずれかの方法で分散トレーニング実験を管理しましょう。

* **単一プロセスをトラッキング**: W&B で rank 0 プロセス（"リーダー" や "コーディネーター" とも呼ばれます）だけをトラッキングします。これは [PyTorch Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP) クラスを用いた分散トレーニングの一般的な方法です。
* **複数プロセスをトラッキング**: 複数プロセスの場合、以下のいずれかの方法が選べます。
   * 各プロセスごとに run を分けて個別にトラッキング。W&B App UI でまとめてグループ化もできます。
   * すべてのプロセスで 1 つの run にまとめてトラッキング。



## 単一プロセスをトラッキング

このセクションでは、rank 0 プロセスで利用可能な値やメトリクスをトラッキングする方法を説明します。この方法は、単一プロセスから取得できるメトリクスだけを追跡したい場合に適しています。代表的なメトリクスには、GPU/CPU の使用率、共有検証セット上での挙動、勾配やパラメータ、代表的なデータサンプルに対する損失値などが含まれます。

rank 0 プロセス内で、[`wandb.init()`]({{< relref "/ref/python/sdk/functions/init" >}}) で run を初期化し、[`wandb.log`]({{< relref "/ref/python/sdk/classes/run/#method-runlog" >}}) で実験結果をログとして記録しましょう。

以下の [サンプル Python スクリプト（`log-ddp.py`）](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) では、PyTorch DDP を用いて 1 台のマシン上の 2 つの GPU でメトリクスをトラッキングする方法を示しています。[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)（`torch.nn` の `DistributedDataParallel`）は分散トレーニングでよく使われるライブラリです。基本的な考え方はどんな分散トレーニング手法にも適応できますが、実装は異なる場合があります。

Python スクリプトの例：
1. `torch.distributed.launch` で複数プロセスを開始
1. `--local_rank` コマンドライン引数で rank を確認
1. rank が 0 の場合のみ、[`train()`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py#L24) 関数内で `wandb` のログ出力を有効化

```python
if __name__ == "__main__":
    # 引数を取得
    args = parse_args()

    if args.local_rank == 0:  # メインプロセスのみ
        # wandb run の初期化
        run = wandb.init(
            entity=args.entity,
            project=args.project,
        )
        # DDP でモデルをトレーニング
        train(args, run)
    else:
        train(args)
```

[単一プロセスからトラッキングしたメトリクスのダッシュボード例](https://wandb.ai/ayush-thakur/DDP/runs/1s56u3hc/system)をぜひご覧ください。

このダッシュボードには、両方の GPU のシステムメトリクス（温度や使用率など）が表示されています。

{{< img src="/images/track/distributed_training_method1.png" alt="GPU metrics dashboard" >}}

ただし、エポックやバッチサイズに対する損失値は、単一の GPU だけから記録されています。

{{< img src="/images/experiments/loss_function_single_gpu.png" alt="Loss function plots" >}}

## 複数プロセスをトラッキング

W&B で複数プロセスをトラッキングするには次のいずれかを選びます：
* 各プロセス毎に run を作成して [個別にトラッキング]({{< relref "distributed-training/#track-each-process-separately" >}})
* [すべてのプロセスを 1 つの run にまとめてトラッキング]({{< relref "distributed-training/#track-all-processes-to-a-single-run" >}})

### 各プロセスを個別にトラッキング

このセクションでは、各プロセスごとに run を作成し分けてトラッキングする手法を説明します。run 内でメトリクスや artifacts などをそれぞれログします。トレーニング終了時には `wandb.Run.finish()` を呼び出して run の終了を明示し、全プロセスが正しく終了できるようにします。

複数の Experiments 間で run を管理するのが難しいと感じる場合は、W&B の初期化時に `group` パラメータ（例: `wandb.init(group='group-name')`）に値を入れて、どの run がどの Experiment に属しているかを紐づけて管理しましょう。Experiments 内でのトレーニングや評価における W&B Runs の管理方法については、[Group Runs]({{< relref "/guides/models/track/runs/grouping.md" >}}) をご覧ください。

{{% alert %}}
**プロセスごとに個別のメトリクスを記録したい場合は、この方法を推奨します。**  
典型例としては、各ノードのデータや予測（データ分布のデバッグ）、またメインノード以外での個々のバッチのメトリクスなどが挙げられます。この手法は、全ノードのシステムメトリクス取得や、メインノードで利用可能なサマリ統計の取得には不要です。
{{% /alert %}}

以下の Python コードスニペットは、W&B 初期化時に group パラメータを指定する方法を示しています。

```python
if __name__ == "__main__":
    # 引数を取得
    args = parse_args()
    # run の初期化
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group="DDP",  # この experiment の全ての run を 1 グループにまとめる
    )
    # DDP でモデルをトレーニング
    train(args, run)

    run.finish()  # run を終了扱いにする
```

W&B App UI で [複数プロセスからトラッキングしたメトリクスのダッシュボード例](https://wandb.ai/ayush-thakur/DDP?workspace=user-noahluna) を確認してください。左サイドバーには 2 つの W&B Runs がグループ化されています。グループ名をクリックすると、その experiment の専用グループページが表示され、各プロセスごとのメトリクスが確認可能です。

{{< img src="/images/experiments/dashboard_grouped_runs.png" alt="Grouped distributed runs" >}}

上記画像は W&B App UI のダッシュボード例です。サイドバーには 2 つの experiment が表示され、1 つは 'null'、もう 1 つ（黄色で囲まれている）は 'DPP' というグループ名です。グループ（Group ドロップダウン）を展開すると、その experiment に紐づく W&B Runs を確認できます。

### すべてのプロセスを 1 つの run にまとめてトラッキング

{{% alert color="secondary"  %}}
`x_` で始まるパラメータ（例：`x_label`）はパブリックプレビュー中です。フィードバックがあれば [W&B リポジトリに GitHub issue](https://github.com/wandb/wandb) を作成してください。
{{% /alert %}}

{{% alert title="要件" %}}
複数プロセスを 1 つの run にまとめてトラッキングするには、以下が必要です。
- W&B Python SDK バージョン `v0.19.9` 以上

- W&B Server v0.68 以上
{{% /alert  %}}

この方法では、プライマリノード 1 台とワーカーノード 1 台以上を利用します。プライマリノード上で W&B run を初期化し、各ワーカーノードでも同じ run ID を使って W&B run を初期化します。トレーニング中は全ワーカーノードが同じ run ID でログを記録し、W&B が各ノードからのメトリクスを集約して W&B App UI に表示します。

プライマリノード内では、[`wandb.init()`]({{< relref "/ref/python/sdk/functions/init" >}}) を使い、`settings` パラメータとして `wandb.Settings` オブジェクト（`wandb.init(settings=wandb.Settings()`）を渡してください。設定内容は以下の通りです。

1. `mode` を `"shared"` に設定（共有モードを有効化）
2. 一意なラベル [`x_label`](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py#L638) を設定。ここで指定した値は、W&B App UI のログやシステムメトリクスで、どのノード由来のデータかを識別するために使われます。未指定の場合は、ホスト名＋ランダムハッシュで自動生成されます。
3. [`x_primary`](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py#L660) を `True` に設定して、これがプライマリノードであることを示します。
4. どの GPU のメトリクスを記録するかを `x_stats_gpu_device_ids`（例：[0,1,2]）にリストで指定可能。指定しない場合、W&B はマシン上のすべての GPU をトラッキングします。

プライマリノードの run ID を控えておきます。各ワーカーノードでこの run ID を利用します。

{{% alert %}}
`x_primary=True` をセットすることで、プライマリノードとワーカーノードの違いを区別します。プライマリノードのみが、設定ファイルやテレメトリなどノード間で共有されるファイルをアップロードします。ワーカーノードはこれらのファイルをアップロードしません。
{{% /alert %}}

各ワーカーノードでは、[`wandb.init()`]({{< relref "/ref/python/sdk/functions/init" >}}) で以下を指定します。
1. `settings` パラメータとして `wandb.Settings` オブジェクト（`wandb.init(settings=wandb.Settings()`）を渡す：
   * `mode` を `"shared"` に設定
   * ユニークな `x_label` を設定（どのノードからのデータか識別するため）。未指定時は自動生成。
   * `x_primary` を `False` に設定（このノードはワーカーであることを示す）
2. プライマリノードの run ID を `id` パラメータに渡す
3. 必要に応じて [`x_update_finish_state`](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py#L772) を `False` に設定。これで非プライマリノードによる [run の状態管理]({{< relref "/guides/models/track/runs/#run-states" >}}) が妨げられ、run の状態が一貫してプライマリノードによって管理されます。

{{% alert %}}
プライマリノードの run ID は、各ワーカーノードのマシン環境変数などで設定しておくと便利です。
{{% /alert %}}

以下のサンプルコードは、複数プロセスを 1 つの run にトラッキングする際の基本構成例です。

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
        x_stats_gpu_device_ids=[0, 1],  # （任意）GPU 0 および 1 のメトリクスだけ追跡
        )
)

# プライマリノードの run ID を控える
# 各ワーカーノードでもこの run ID を利用
run_id = run.id

# ワーカーノードで run を初期化（プライマリノードの run ID を使う）
run = wandb.init(
	settings=wandb.Settings(x_label="rank_1", mode="shared", x_primary=False),
	id=run_id,
)

# 別のワーカーノードの例（同じくプライマリノードの run ID を使う）
run = wandb.init(
	settings=wandb.Settings(x_label="rank_2", mode="shared", x_primary=False),
	id=run_id,
)
```

実際の運用では、各ワーカーノードが別のマシン上で動くこともあります。

{{% alert %}}
[Distributed Training with Shared Mode](https://wandb.ai/dimaduev/simple-cnn-ddp/reports/Distributed-Training-with-Shared-Mode--VmlldzoxMTI0NTE1NA) レポートでは、GKE 環境のマルチノード & マルチ GPU Kubernetes クラスター上での分散学習のエンドツーエンド例を紹介しています。
{{% /alert %}}

multi node プロセスからのコンソールログをその run を含む Project で見るには：

1. 該当する Project ページに移動
2. 左サイドバーで **Runs** タブをクリック
3. 見たい run をクリック
4. 左サイドバーで **Logs** タブをクリック

UI の検索バーで `x_label` の値によるログのフィルタリングが可能です。下記のように `rank0`, `rank1`, `rank2`, `rank3`, `rank4`, `rank5`, `rank6` などを設定していれば、その値でフィルタできます。

{{< img src="/images/track/multi_node_console_logs.png" alt="Multi-node console logs" >}}

詳細は [Console logs]({{< relref "/guides/models/app/console-logs/" >}}) をご覧ください。

W&B は全ノードからのシステムメトリクスを集約し、W&B App UI で閲覧できます。以下は複数ノードからのサンプルダッシュボードです。それぞれのノードは `x_label`（例：`rank_0`, `rank_1`, `rank_2`）で一意に識別されます。

{{< img src="/images/track/multi_node_system_metrics.png" alt="Multi-node system metrics" >}}

ラインプロットパネルのカスタマイズ方法は [Line plots]({{< relref "/guides/models/app/features/panels/line-plot/" >}}) をご覧ください。

## ユースケース例

以下のコード例は、応用的な分散処理ユースケースでよく利用されるシナリオを示しています。

### プロセスのスポーン

spawn したプロセスで run を開始する場合、メイン関数内で `wandb.setup()` を利用します。

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

### run の共有

run オブジェクトを引数として渡すことで、プロセス間で run を共有できます。

```python
def do_work(run):
    with wandb.init() as run:
        run.log(dict(this=1))

def main():
    run = wandb.init()
    p = mp.Process(target=do_work, kwargs=dict(run=run))
    p.start()
    p.join()
    run.finish()  # run を終了扱いにする


if __name__ == "__main__":
    main()
```

W&B のログの順序は保証されません。必要に応じて同期処理はスクリプト作者側で制御してください。


## トラブルシューティング

W&B と分散トレーニングを併用する際によくある問題は主に 2 つあります：

1. **トレーニング開始時のハングアップ** - `wandb` のマルチプロセスが分散トレーニングのマルチプロセスと干渉し、`wandb` プロセスがハングすることがあります。
2. **トレーニング終了時のハングアップ** - `wandb` プロセスが終了タイミングを検出できず、ジョブがハングすることがあります。Python スクリプトの最後で `wandb.Run.finish()` API を呼び出して run の終了を明示してください。`wandb.Run.finish()` API を使うことでアップロードとプロセス終了が確実に行われます。
分散ジョブの信頼性向上のため、`wandb service` コマンドの利用を推奨します。これらのトレーニング問題は、wandb service が利用できない SDK バージョンで特に多くみられます。

### W&B Service を有効化

ご利用中の W&B SDK バージョンによっては、W&B Service がデフォルトで有効になっています。

#### W&B SDK 0.13.0 以降

SDK バージョン `0.13.0` 以上では W&B Service はデフォルトで有効です。

#### W&B SDK 0.12.5 以上

W&B SDK バージョン 0.12.5 以降で W&B Service を有効にするには、メイン関数内で `wandb.require` メソッドに `"service"` を渡してください：

```python
if __name__ == "__main__":
    main()


def main():
    wandb.require("service")
    # 以降にスクリプトの本体を書く
```

最新バージョンの利用を強く推奨します。

**W&B SDK 0.12.4 以下の場合**

SDK バージョン 0.12.4 以下の場合は、`WANDB_START_METHOD` 環境変数に `"thread"` を設定してマルチスレッドを使用してください。