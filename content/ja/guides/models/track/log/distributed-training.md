---
title: Log distributed training experiments
description: W&B を使用して、複数の GPU を使用した分散型トレーニング の 実験管理 を ログ 記録します。
menu:
  default:
    identifier: ja-guides-models-track-log-distributed-training
    parent: log-objects-and-media
---

分散トレーニングでは、モデルは複数の GPU を並行して使用してトレーニングされます。W&B は、分散トレーニング の 実験管理 を追跡するための 2 つのパターンをサポートしています。

1. **単一 プロセス**: W&B ([`wandb.init`]({{< relref path="/ref//python/init.md" lang="ja" >}})) を初期化し、単一の プロセス から 実験 ([`wandb.log`]({{< relref path="/ref//python/log.md" lang="ja" >}})) を ログ 記録します。これは、[PyTorch Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP) クラスを使用した分散トレーニング の 実験 を ログ 記録するための一般的なソリューションです。場合によっては、マルチプロセッシング キュー (または別の通信プリミティブ) を使用して、他の プロセス からメインの ログ 記録 プロセス にデータを送り込む ユーザー もいます。
2. **多数の プロセス**: W&B ([`wandb.init`]({{< relref path="/ref//python/init.md" lang="ja" >}})) を初期化し、すべての プロセス で 実験 ([`wandb.log`]({{< relref path="/ref//python/log.md" lang="ja" >}})) を ログ 記録します。各 プロセス は、事実上別の 実験 です。W&B を初期化する際に `group` パラメータ (`wandb.init(group='group-name')`) を使用して、共有 実験 を定義し、 ログ 記録された 値 を W&B App UI にまとめて グループ化します。

以下の例では、単一 マシン 上の 2 つの GPU で PyTorch DDP を使用して、W&B で メトリクス を追跡する方法を示します。[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (`torch.nn` の `DistributedDataParallel`) は、分散トレーニング 用の一般的な ライブラリ です。基本的な原則は、あらゆる分散トレーニング 設定に適用されますが、実装の詳細は異なる場合があります。

{{% alert %}}
これらの例の背後にある コード を W&B GitHub の examples リポジトリ ([こちら](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-ddp)) で確認してください。特に、単一 プロセス および 多数 プロセス の メソッド を実装する方法については、[`log-dpp.py`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) Python スクリプトを参照してください。
{{% /alert %}}

### 方法 1: 単一 プロセス

この方法では、ランク 0 の プロセス のみを追跡します。この方法を実装するには、W&B (`wandb.init`) を初期化し、W&B Run を開始して、ランク 0 の プロセス 内で メトリクス (`wandb.log`) を ログ 記録します。この方法はシンプルで堅牢ですが、他の プロセス から モデル の メトリクス (たとえば、バッチからの 損失 値 または 入力) を ログ 記録しません。使用量 や メモリ などの システム メトリクス は、その情報がすべての プロセス で利用できるため、すべての GPU に対して ログ 記録されます。

{{% alert %}}
**この方法を使用して、単一の プロセス から利用可能な メトリクス のみを追跡します**。一般的な例としては、GPU/CPU の使用率、共有 検証セット での 振る舞い 、 勾配 と パラメータ 、および代表的な データ 例での 損失 値 などがあります。
{{% /alert %}}

[サンプル Python スクリプト (`log-ddp.py`)](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) 内で、ランクが 0 かどうかを確認します。これを行うには、まず `torch.distributed.launch` で複数の プロセス を 起動します。次に、`--local_rank` コマンドライン 引数 でランクを確認します。ランクが 0 に設定されている場合、[`train()`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py#L24) 関数で `wandb` ログ 記録を条件付きで設定します。Python スクリプト内で、次のチェックを使用します。

```python
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

W&B App UI を調べて、単一の プロセス から追跡された メトリクス の [ダッシュボード 例](https://wandb.ai/ayush-thakur/DDP/runs/1s56u3hc/system) を表示します。ダッシュボード には、両方の GPU で追跡された 温度 や 使用率 などの システム メトリクス が表示されます。

{{< img src="/images/track/distributed_training_method1.png" alt="" >}}

ただし、 エポック と バッチサイズ の関数としての 損失 値 は、単一の GPU からのみ ログ 記録されました。

{{< img src="/images/experiments/loss_function_single_gpu.png" alt="" >}}

### 方法 2: 多数の プロセス

この方法では、ジョブ内の各 プロセス を追跡し、各 プロセス から `wandb.init()` と `wandb.log()` を個別に呼び出します。すべての プロセス が適切に終了するように、トレーニング の最後に `wandb.finish()` を呼び出すことをお勧めします。これにより、run が完了したことを示します。

この方法により、より多くの情報が ログ 記録にアクセスできるようになります。ただし、複数の W&B Runs が W&B App UI に 報告 されることに注意してください。複数の 実験 にわたって W&B Runs を追跡することが難しい場合があります。これを軽減するには、W&B を初期化する際に group パラメータ に 値 を指定して、どの W&B Run が特定の 実験 に属しているかを追跡します。トレーニング と 評価 の W&B Runs を 実験 で追跡する方法の詳細については、[Run のグループ化]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) を参照してください。

{{% alert %}}
**個々の プロセス から メトリクス を追跡する場合は、この方法を使用してください**。一般的な例としては、各 ノード 上の データ と 予測 (データ 分布 の デバッグ 用) や、メイン ノード 外の個々の バッチ 上の メトリクス などがあります。この方法は、すべての ノード から システム メトリクス を取得したり、メイン ノード で利用可能な概要 統計 を取得したりするために必要ではありません。
{{% /alert %}}

次の Python コード スニペット は、W&B を初期化するときに group パラメータ を設定する方法を示しています。

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

W&B App UI を調べて、複数の プロセス から追跡された メトリクス の [ダッシュボード 例](https://wandb.ai/ayush-thakur/DDP?workspace=user-noahluna) を表示します。左側のサイドバーに 2 つの W&B Runs が グループ化 されていることに注意してください。グループ をクリックして、 実験 専用のグループ ページ を表示します。専用のグループ ページ には、各 プロセス からの メトリクス が個別に表示されます。

{{< img src="/images/experiments/dashboard_grouped_runs.png" alt="" >}}

上記の画像は、W&B App UI ダッシュボード を示しています。サイドバーには、2 つの 実験 が表示されます。1 つは「null」というラベルが付いており、2 つ目 (黄色のボックスで囲まれています) は「DPP」と呼ばれています。グループ を展開すると (グループ ドロップダウン を選択)、その 実験 に関連付けられている W&B Runs が表示されます。

### W&B Service を使用して、一般的な分散トレーニング の問題を回避する

W&B と分散トレーニング を使用する際に発生する可能性のある一般的な問題が 2 つあります。

1. **トレーニング の開始時にハングする** - `wandb` マルチプロセッシング が分散トレーニング からの マルチプロセッシング に干渉すると、`wandb` プロセス がハングする可能性があります。
2. **トレーニング の最後にハングする** - `wandb` プロセス がいつ終了する必要があるかを認識していない場合、トレーニング ジョブ がハングする可能性があります。Python スクリプト の最後に `wandb.finish()` API を呼び出して、Run が完了したことを W&B に伝えます。wandb.finish() API は、データの アップロード を終了し、W&B を終了させます。

分散ジョブ の信頼性を向上させるために、`wandb service` を使用することをお勧めします。上記のトレーニング の問題はどちらも、wandb service が利用できない W&B SDK の バージョン でよく見られます。

### W&B Service を有効にする

W&B SDK の バージョン によっては、W&B Service がデフォルトで有効になっている場合があります。

#### W&B SDK 0.13.0 以降

W&B Service は、W&B SDK `0.13.0` 以降の バージョン ではデフォルトで有効になっています。

#### W&B SDK 0.12.5 以降

Python スクリプト を変更して、W&B SDK バージョン 0.12.5 以降で W&B Service を有効にします。`wandb.require` メソッド を使用し、メイン関数内で 文字列 `"service"` を渡します。

```python
if __name__ == "__main__":
    main()


def main():
    wandb.require("service")
    # rest-of-your-script-goes-here
```

最適なエクスペリエンスを得るには、最新 バージョン に アップグレード することをお勧めします。

**W&B SDK 0.12.4 以前**

W&B SDK バージョン 0.12.4 以前を使用している場合は、マルチスレッド を代わりに使用するために、`WANDB_START_METHOD` 環境 変数 を `"thread"` に設定します。

### マルチプロセッシング の ユースケース 例

次の コード スニペット は、高度な分散 ユースケース の一般的な方法を示しています。

#### プロセス の スポーン

スポーンされた プロセス で W&B Run を開始する場合は、メイン関数で `wandb.setup()` メソッド を使用します。

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

#### W&B Run を共有する

W&B Run オブジェクト を 引数 として渡して、 プロセス 間で W&B Runs を共有します。

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
ログ 記録の順序は保証されないことに注意してください。同期は スクリプト の作成者が行う必要があります。
{{% /alert %}}
