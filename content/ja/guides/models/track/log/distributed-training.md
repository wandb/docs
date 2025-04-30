---
title: 分散トレーニング実験をログする
description: W&B を使用して、複数の GPU を用いた分散トレーニング実験をログする。
menu:
  default:
    identifier: ja-guides-models-track-log-distributed-training
    parent: log-objects-and-media
---

分散トレーニングでは、複数の GPU を使ってモデルが並列にトレーニングされます。W&B は、分散トレーニング実験をトラッキングするための2つのパターンをサポートしています。

1. **ワンプロセス**: 単一のプロセスから W&B を初期化し（[`wandb.init`]({{< relref path="/ref//python/init.md" lang="ja" >}})）、実験をログします（[`wandb.log`]({{< relref path="/ref//python/log.md" lang="ja" >}})）。これは [PyTorch Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel)（DDP）クラスを使った分散トレーニング実験のログに一般的なソリューションです。ユーザーは他のプロセスからメインのロギングプロセスにデータを送るために、多重処理キュー（または他の通信プリミティブ）を使用することもあります。
2. **多数のプロセス**: 各プロセスで W&B を初期化し（[`wandb.init`]({{< relref path="/ref//python/init.md" lang="ja" >}})）、実験をログします（[`wandb.log`]({{< relref path="/ref//python/log.md" lang="ja" >}})）。各プロセスは実質的に別々の実験です。W&B を初期化する際に、`group` パラメータを使用して共有実験を定義し、W&B App UI のログした値を一緒にグループ化します。

次に示す例は、PyTorch DDP を使って単一のマシン上で2つの GPU でメトリクスを W&B でトラッキングする方法を示しています。[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)（`torch.nn` の `DistributedDataParallel`）は、分散トレーニングのための人気のあるライブラリです。基本的な原則はどの分散トレーニングセットアップにも適用されますが、実装の詳細は異なる場合があります。

{{% alert %}}
これらの例の背後にあるコードを [W&B GitHub examples リポジトリ](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-ddp) で探してください。特に、1つのプロセスと多くのプロセスメソッドを実装する方法については、[`log-dpp.py`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) の Python スクリプトを参照してください。
{{% /alert %}}

### 方法 1: ワンプロセス

この方法では、ランク 0 のプロセスのみをトラッキングします。この方法を実装するには、ランク 0 のプロセス内で W&B を初期化し（`wandb.init`）、W&B Run を開始し、メトリクスをログ（`wandb.log`）します。この方法はシンプルで堅牢ですが、他のプロセスからモデルメトリクス（例えば、ロス値や各バッチからの入力）をログしません。使用状況やメモリなどのシステムメトリクスは、すべての GPU に利用可能な情報であるため、引き続きログされます。

{{% alert %}}
**単一のプロセスで利用可能なメトリクスのみをトラッキングする場合、この方法を使用してください。** 典型的な例には、GPU/CPU 使用率、共有 validation set 上の挙動、勾配とパラメータ、代表的なデータ例上の損失値が含まれます。
{{% /alert %}}

[サンプル Python スクリプト (`log-ddp.py`)](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) では、ランクが 0 かどうかを確認します。そのためには、まず `torch.distributed.launch` を使って複数のプロセスを開始します。次に、`--local_rank` コマンドライン引数を使用してランクを確認します。ランクが 0 に設定されている場合、[`train()`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py#L24) 関数内で条件付きで `wandb` ロギングを設定します。Python スクリプト内では、次のように確認します。

```python
if __name__ == "__main__":
    # 引数を取得
    args = parse_args()

    if args.local_rank == 0:  # メインプロセスでのみ
        # wandb run を初期化
        run = wandb.init(
            entity=args.entity,
            project=args.project,
        )
        # DDP でモデルをトレーニング
        train(args, run)
    else:
        train(args)
```

W&B App UI を探索して、単一プロセスからトラッキングされたメトリクスの [ダッシュボードの例](https://wandb.ai/ayush-thakur/DDP/runs/1s56u3hc/system) をご覧ください。ダッシュボードは、両方の GPU に対してトラッキングされた温度や使用率などのシステムメトリクスを表示します。

{{< img src="/images/track/distributed_training_method1.png" alt="" >}}

しかし、エポックとバッチサイズの関数としてのロス値は、単一の GPU からのみログされました。

{{< img src="/images/experiments/loss_function_single_gpu.png" alt="" >}}

### 方法 2: 多数のプロセス

この方法では、ジョブ内の各プロセスをトラッキングし、各プロセスから個別に `wandb.init()` と `wandb.log()` を呼び出します。トレーニングの終了時には `wandb.finish()` を呼び出して、run が完了したことを示し、すべてのプロセスが正常に終了するようにすることをお勧めします。

この方法では、さらに多くの情報がログにアクセス可能になりますが、W&B App UI に複数の W&B Runs が報告されます。複数の実験にわたって W&B Runs を追跡するのが困難になる可能性があります。これを軽減するために、W&B を初期化する際に `group` パラメータに値を与えて、どの W&B Run がどの実験に属しているかを追跡します。実験でのトレーニングと評価の W&B Runs の追跡方法の詳細については、[Group Runs]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) を参照してください。

{{% alert %}}
**個々のプロセスからメトリクスをトラッキングしたい場合はこの方法を使用してください。** 典型的な例には、各ノードでのデータと予測（データ分散のデバッグ用）やメインノードの外側での個々のバッチのメトリクスが含まれます。この方法は、すべてのノードからのシステムメトリクスやメインノードで利用可能な要約統計データを取得するために必要ありません。
{{% /alert %}}

以下の Python コードスニペットは、W&B を初期化する際に `group` パラメータを設定する方法を示しています。

```python
if __name__ == "__main__":
    # 引数を取得
    args = parse_args()
    # run を初期化
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group="DDP",  # 実験のすべての run を1つのグループに
    )
    # DDP でモデルをトレーニング
    train(args, run)
```

W&B App UI を探索して、複数のプロセスからトラッキングされたメトリクスの [ダッシュボードの例](https://wandb.ai/ayush-thakur/DDP?workspace=user-noahluna) をご覧ください。左側のサイドバーに 2 つの W&B Runs が組み合わされたものが示されています。グループをクリックして、その実験専用のグループページを表示します。専用のグループページには、各プロセスから別々にログされたメトリクスが表示されます。

{{< img src="/images/experiments/dashboard_grouped_runs.png" alt="" >}}

前の画像は W&B App UI ダッシュボードを示しています。サイドバーには2つの実験が表示されています。1つは「null」とラベル付けされ、黄色のボックスで囲まれた2つ目は「DPP」と呼ばれます。グループを展開すると（[Group] ドロップダウンを選択）、その実験に関連する W&B Runs を見ることができます。

### 共通の分散トレーニングの問題を避けるために W&B Service を使用

W&B と分散トレーニングを使用する場合、2つの一般的な問題に遭遇することがあります。

1. **トレーニングの開始時のハング** - `wandb` プロセスが、分散トレーニングからの多重処理と干渉するためにハングすることがあります。
2. **トレーニングの終了時のハング** - トレーニングジョブが、`wandb` プロセスがいつ終了する必要があるかを知らない場合、ハングすることがあります。Python スクリプトの最後に `wandb.finish()` API を呼び出して、W&B に Run が終了したことを通知します。wandb.finish() API はデータのアップロードを完了し、W&B の終了を引き起こします。

`wandb service` を使用して、分散ジョブの信頼性を向上させることをお勧めします。上記のトレーニングの問題は、wandb service が利用できない W&B SDK のバージョンで一般的に見られます。

### W&B Service の有効化

お使いのバージョンの W&B SDK に応じて、すでにデフォルトで W&B Service が有効になっているかもしれません。

#### W&B SDK 0.13.0 以上

W&B SDK バージョン `0.13.0` 以上のバージョンでは、W&B Service がデフォルトで有効です。

#### W&B SDK 0.12.5 以上

W&B SDK バージョン 0.12.5 以上の場合は、Python スクリプトを修正して W&B Service を有効にします。`wandb.require` メソッドを使用し、メイン関数内で文字列 `"service"` を渡します。

```python
if __name__ == "__main__":
    main()


def main():
    wandb.require("service")
    # スクリプトの残りがここに来る
```

最適な体験のために、最新バージョンへのアップグレードをお勧めします。

**W&B SDK 0.12.4 以下**

W&B SDK バージョン 0.12.4 以下を使用する場合は、マルチスレッドを代わりに使用するために、`WANDB_START_METHOD` 環境変数を `"thread"` に設定します。

### マルチプロセスの例々

以下のコードスニペットは、高度な分散ユースケースの一般的なメソッドを示しています。

#### プロセスの生成

ワークスレッドを生成するプロセス内で W&B Run を開始する場合は、メイン関数で `wandb.setup()` メソッドを使用します。

```python
import multiprocessing as mp


def do_work(n):
    run = wandb.init(config=dict(n=n))
    run.log(dict(this=n * n))


def main():
    wandb.setup()
    pool = mp.Pool(processes=4)
    pool.map(do_work, range(4))


if __name__ == "__main__":
    main()
```

#### W&B Run の共有

W&B Run オブジェクトを引数として渡して、プロセス間で W&B Runs を共有します。

```python
def do_work(run):
    run.log(dict(this=1))


def main():
    run = wandb.init()
    p = mp.Process(target=do_work, kwargs=dict(run=run))
    p.start()
    p.join()


if __name__ == "__main__":
    main()
```

{{% alert %}}
記録の順序は保証できないことに注意してください。同期はスクリプトの作成者が行う必要があります。
{{% /alert %}}