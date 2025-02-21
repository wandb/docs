---
title: Log distributed training experiments
description: W&B を使用して、複数の GPU を使用した分散トレーニング 実験 の ログ を記録します。
menu:
  default:
    identifier: ja-guides-models-track-log-distributed-training
    parent: log-objects-and-media
---

分散トレーニングでは、モデルは複数の GPU を並行して使用してトレーニングされます。W&B は、分散トレーニングの実験を追跡するための 2 つのパターンをサポートしています。

1. **単一プロセス**: 単一のプロセスから W&B ( [`wandb.init`]({{< relref path="/ref//python/init.md" lang="ja" >}})) を初期化し、実験 ( [`wandb.log`]({{< relref path="/ref//python/log.md" lang="ja" >}})) を記録します。これは、[PyTorch Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP) クラスを使用した分散トレーニング実験を記録するための一般的なソリューションです。場合によっては、マルチプロセッシングキュー (または別の通信プリミティブ) を使用して、他のプロセスからメインのロギングプロセスにデータを送ります。
2. **多プロセス**: すべてのプロセスで W&B ( [`wandb.init`]({{< relref path="/ref//python/init.md" lang="ja" >}})) を初期化し、実験 ( [`wandb.log`]({{< relref path="/ref//python/log.md" lang="ja" >}})) を記録します。各プロセスは事実上、個別の実験です。W&B を初期化するときに `group` パラメータ (`wandb.init(group='group-name')`) を使用して、共有実験を定義し、記録された値を W&B App UI でグループ化します。

以下の例では、単一のマシン上の 2 つの GPU で PyTorch DDP を使用して、W&B でメトリクスを追跡する方法を示します。[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (`torch.nn` の`DistributedDataParallel`) は、分散トレーニング用の一般的なライブラリです。基本的な原則は、あらゆる分散トレーニング設定に適用されますが、実装の詳細は異なる場合があります。

{{% alert %}}
これらの例の背後にあるコードは、W&B GitHub examples リポジトリ ([こちら](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-ddp)) で確認してください。特に、1 つのプロセスと多プロセスメソッドを実装する方法については、[`log-dpp.py`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) Python スクリプトを参照してください。
{{% /alert %}}

### 方法 1: 単一プロセス

この方法では、ランク 0 のプロセスのみを追跡します。この方法を実装するには、W&B (`wandb.init`) を初期化し、W&B Run を開始し、ランク 0 のプロセス内でメトリクス (`wandb.log`) を記録します。この方法はシンプルで堅牢ですが、他のプロセスからのモデルメトリクス (たとえば、バッチからの損失値や入力) は記録されません。使用状況やメモリなど、システムメトリクスは、その情報がすべてのプロセスで利用できるため、すべての GPU で記録されます。

{{% alert %}}
**この方法は、単一のプロセスから利用できるメトリクスのみを追跡するために使用します**。一般的な例としては、GPU / CPU の使用率、共有検証セットでの振る舞い、勾配とパラメータ、代表的なデータ例での損失値などがあります。
{{% /alert %}}

[サンプル Python スクリプト (`log-ddp.py`)](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) 内で、ランクが 0 であるかどうかを確認します。これを行うには、まず `torch.distributed.launch` で複数のプロセスを起動します。次に、`--local_rank` コマンドライン引数でランクを確認します。ランクが 0 に設定されている場合は、[`train()`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py#L24) 関数で条件付きで `wandb` ロギングを設定します。Python スクリプト内では、次のチェックを使用します。

```python showLineNumbers
if __name__ == "__main__":
    # Get args
    args = parse_args()

    if args.local_rank == 0:  # only on main process
        # Initialize wandb run
        run = wandb.init(
            entity=args.entity,
            project=args.project,
        )
        # Train model with DDP
        train(args, run)
    else:
        train(args)
```

W&B App UI を調べて、単一のプロセスから追跡されたメトリクスの [ダッシュボードの例](https://wandb.ai/ayush-thakur/DDP/runs/1s56u3hc/system) を表示します。ダッシュボードには、両方の GPU で追跡された温度や使用率などのシステムメトリクスが表示されます。

{{< img src="/images/track/distributed_training_method1.png" alt="" >}}

ただし、エポックとバッチサイズを関数とする損失値は、単一の GPU からのみ記録されました。

{{< img src="/images/experiments/loss_function_single_gpu.png" alt="" >}}

### 方法 2: 多数のプロセス

この方法では、ジョブ内の各プロセスを追跡し、各プロセスから `wandb.init()` と `wandb.log()` を個別に呼び出します。トレーニングの最後に `wandb.finish()` を呼び出して、すべてのプロセスが適切に終了するように、run が完了したことを示すことをお勧めします。

この方法により、ロギングにアクセスできる情報が増えます。ただし、複数の W&B Runs が W&B App UI で報告されることに注意してください。複数の Experiments で W&B Runs を追跡することが難しい場合があります。これを軽減するには、W&B を初期化するときに group パラメータに値を指定して、どの W&B Run が特定の Experiment に属しているかを追跡します。Experiments でトレーニングと評価の W&B Runs を追跡する方法の詳細については、[Runs のグループ化]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) を参照してください。

{{% alert %}}
**個々のプロセスからメトリクスを追跡する場合は、この方法を使用します**。一般的な例としては、各ノードのデータと予測 (データ分散をデバッグするため)、およびメインノード外の個々のバッチのメトリクスがあります。この方法は、すべてのノードからシステムメトリクスを取得したり、メインノードで利用可能な要約統計を取得したりするために必要ではありません。
{{% /alert %}}

次の Python コードスニペットは、W&B を初期化するときに group パラメータを設定する方法を示しています。

```python
if __name__ == "__main__":
    # Get args
    args = parse_args()
    # Initialize run
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group="DDP",  # all runs for the experiment in one group
    )
    # Train model with DDP
    train(args, run)
```

W&B App UI を調べて、複数のプロセスから追跡されたメトリクスの [ダッシュボードの例](https://wandb.ai/ayush-thakur/DDP?workspace=user-noahluna) を表示します。左側のサイドバーに 2 つの W&B Runs がグループ化されていることに注意してください。グループをクリックして、Experiment の専用グループページを表示します。専用グループページには、各プロセスのメトリクスが個別に表示されます。

{{< img src="/images/experiments/dashboard_grouped_runs.png" alt="" >}}

上の画像は、W&B App UI ダッシュボードを示しています。サイドバーには、2 つの Experiments が表示されています。1 つは「null」というラベルが付けられ、2 つ目は「DPP」というラベルが付けられています (黄色のボックスで囲まれています)。グループを展開すると (グループドロップダウンを選択)、その Experiment に関連付けられている W&B Runs が表示されます。

### W&B Service を使用して、一般的な分散トレーニングの問題を回避する

W&B と分散トレーニングを使用する場合、発生する可能性のある一般的な問題が 2 つあります。

1. **トレーニングの開始時にハングする** - `wandb` プロセスは、`wandb` マルチプロセッシングが分散トレーニングからのマルチプロセッシングと干渉する場合にハングする可能性があります。
2. **トレーニングの最後にハングする** - `wandb` プロセスがいつ終了する必要があるかを知らない場合、トレーニングジョブがハングする可能性があります。Python スクリプトの最後に `wandb.finish()` API を呼び出して、Run が完了したことを W&B に伝えます。wandb.finish() API は、データのアップロードを完了し、W&B を終了させます。

分散ジョブの信頼性を向上させるために、`wandb service` を使用することをお勧めします。上記のトレーニングの問題はどちらも、wandb service が利用できない W&B SDK のバージョンでよく見られます。

### W&B Service を有効にする

W&B SDK のバージョンによっては、W&B Service がデフォルトで有効になっている場合があります。

#### W&B SDK 0.13.0 以降

W&B Service は、W&B SDK のバージョン `0.13.0` 以降ではデフォルトで有効になっています。

#### W&B SDK 0.12.5 以降

Python スクリプトを変更して、W&B SDK バージョン 0.12.5 以降の W&B Service を有効にします。`wandb.require` メソッドを使用し、メイン関数内で文字列 `"service"` を渡します。

```python
if __name__ == "__main__":
    main()


def main():
    wandb.require("service")
    # rest-of-your-script-goes-here
```

最適なエクスペリエンスを得るには、最新バージョンにアップグレードすることをお勧めします。

**W&B SDK 0.12.4 以前**

W&B SDK バージョン 0.12.4 以前を使用する場合は、`WANDB_START_METHOD` 環境変数を `"thread"` に設定して、代わりにマルチスレッドを使用します。

### マルチプロセッシングのユースケースの例

次のコードスニペットは、高度な分散ユースケースの一般的な方法を示しています。

#### プロセスの生成

生成されたプロセスで W&B Run を開始する場合は、メイン関数で `wandb.setup()[line 8]` メソッドを使用します。

```python showLineNumbers
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

```python showLineNumbers
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
ロギングの順序は保証されないことに注意してください。同期は、スクリプトの作成者が行う必要があります。
{{% /alert %}}
