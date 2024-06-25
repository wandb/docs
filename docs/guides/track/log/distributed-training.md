---
description: W&B を使用して、複数の GPU を用いた分散トレーニング実験をログします。
displayed_sidebar: default
---


# 分散トレーニング実験をログする

<head>
  <title>分散トレーニング実験をログする</title>
</head>

分散トレーニングでは、複数のGPUを並行して使用してモデルをトレーニングします。W&Bは分散トレーニング実験をトラックするために、以下の2つのパターンをサポートしています：

1. **ワンプロセス**: 単一のプロセスでW&Bを初期化して([`wandb.init`](../../../ref//python/init.md))、実験をログします([`wandb.log`](../../../ref//python/log.md))。これは、[PyTorch Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP) クラスを使用した分散トレーニング実験をログするための一般的な方法です。場合によっては、ユーザーは他のプロセスからデータをマルチプロセッシングキュー（または他の通信原始）を使ってメインのログプロセスに集めます。
2. **マルチプロセス**: 各プロセスでW&Bを初期化して([`wandb.init`](../../../ref//python/init.md))、実験をログします([`wandb.log`](../../../ref//python/log.md))。各プロセスは実質的に別々の実験となります。W&Bを初期化する際に`group`パラメータを使用することで、共通の実験を定義し、ログされた値をW&BアプリUI内でグループ化します (`wandb.init(group='group-name')`)。

次の例では、PyTorch DDPを使用して単一のマシン上の2つのGPUでメトリクスをトラックする方法を示します。[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (`torch.nn`内の`DistributedDataParallel`)は、分散トレーニングのための一般的なライブラリです。基本的な原則は他のいかなる分散トレーニングのセットアップにも適用されますが、実装の詳細は異なるかもしれません。

:::info
これらの例のコードはW&B GitHubのexamplesリポジトリで[ここ](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-ddp)から確認できます。具体的には、ワンプロセスとマルチプロセスの方法を実装する方法については、[`log-ddp.py`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) Pythonスクリプトを参照してください。
:::

### 方法1: ワンプロセス

この方法では、ランク0のプロセスのみをトラックします。この方法を実装するには、W&Bを初期化し (`wandb.init`)、W&B Runを開始し、ランク0プロセス内でメトリクスをログします (`wandb.log`)。この方法はシンプルで頑強ですが、他のプロセスからのモデルメトリクス（例えば、ロス値やバッチからの入力）をログしません。システムメトリクス（使用状況やメモリなど）は全GPUでログされます。

:::info
**単一のプロセスから利用可能なメトリクスをトラックするためにこの方法を使ってください**。典型的な例には、GPU/CPUの利用率、共有検証セットでの振る舞い、勾配とパラメータ、代表的なデータ例でのロス値などがあります。
:::

[サンプルPythonスクリプト (`log-ddp.py`)](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py)では、ランクが0であることを確認します。まず、`torch.distributed.launch`を用いて複数のプロセスを起動し、次に`--local_rank`コマンドライン引数を用いてランクを確認します。ランクが0に設定されている場合、`wandb`のログを条件付きでセットアップします。Pythonスクリプト内で次のようなチェックを行います：

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

W&BアプリUIを探索して、単一プロセスからトラックされたメトリクスの[例のダッシュボード](https://wandb.ai/ayush-thakur/DDP/runs/1s56u3hc/system)を参照してください。このダッシュボードは、両方のGPUでトラックされた温度や利用状況などのシステムメトリクスを表示します。

![](/images/track/distributed_training_method1.png)

しかし、エポックとバッチサイズによるロス値は単一のGPUからのみログされました。

![](/images/experiments/loss_function_single_gpu.png)

### 方法2: マルチプロセス

この方法では、ジョブ内の各プロセスをトラックし、各プロセスから個別に`wandb.init()`および`wandb.log()`を呼び出します。トレーニング終了時に`wandb.finish()`を呼び出すことをお勧めします。これはRunが完了したことを示し、すべてのプロセスが適切に終了することを保証します。

この方法では、より多くの情報がログにアクセス可能になります。ただし、W&BアプリUIでは複数のW&B Runsが報告されます。複数の実験にわたってW&B Runsを追跡するのは難しいかもしれません。これを軽減するために、W&Bを初期化する際にgroupパラメータに値を提供し、どのW&B Runがどの実験に属するかを追跡してください。実験内でトレーニングと評価のW&B Runsを追跡する方法の詳細については、[Group Runs](../../runs/grouping.md)を参照してください。

:::info
**個々のプロセスからのメトリクスをトラックしたい場合にこの方法を使ってください**。典型的な例には、各ノード上でのデータと予測（データ分配のデバッグ用）や、メインノード外の個々のバッチ上のメトリクスがあります。この方法は、すべてのノードからのシステムメトリクスや、メインノードで利用可能なサマリ統計を取得するためには必要ありません。
:::

次のPythonコードスニペットでは、W&Bを初期化する際にgroupパラメータを設定する方法を示します：

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

W&BアプリUIを探索して、複数のプロセスからトラックされたメトリクスの[例のダッシュボード](https://wandb.ai/ayush-thakur/DDP?workspace=user-noahluna)を参照してください。サイドバーには2つのW&B Runsがグループ化されて表示されています。グループをクリックして、その実験の専用グループページを表示します。専用グループページには、各プロセスからのメトリクスが個別に表示されます。

![](/images/experiments/dashboard_grouped_runs.png)

前述の画像では、W&BアプリUIのダッシュボードを示しています。サイドバーには2つの実験が表示されています。1つは 'null' とラベル付けされており、もう1つは（黄色のボックスで囲まれた） 'DPP' と呼ばれています。グループを展開すると（Groupドロップダウンを選択）、その実験に関連するW&B Runsが表示されます。

### W&Bサービスを使用して、分散トレーニングの一般的な問題を回避する

W&Bおよび分散トレーニングを使用する際に遭遇する可能性のある一般的な問題が2つあります：

1. **トレーニングの開始時にハングする** - `wandb`プロセスが分散トレーニングのマルチプロセッシングと干渉する場合、`wandb` プロセスがハングする可能性があります。
2. **トレーニングの終了時にハングする** - `wandb`プロセスが終了する必要があることを認識しない場合、トレーニングジョブがハングする可能性があります。Pythonスクリプトの最後に`wandb.finish()` APIを呼び出して、W&BにRunが終了したことを通知してください。`wandb.finish()` APIはデータのアップロードを完了し、W&Bの終了をトリガーします。

分散ジョブの信頼性を向上させるためには、`wandb service`を使用することをお勧めします。上記のトレーニングの問題は、`wandb service`が利用できないW&B SDKのバージョンで一般的に見られます。

### W&Bサービスを有効化

お使いのW&B SDKのバージョンに応じて、すでにデフォルトでW&Bサービスが有効になっている場合があります。

#### W&B SDK 0.13.0以降

W&B SDK `0.13.0`以降のバージョンでは、デフォルトでW&Bサービスが有効になっています。

#### W&B SDK 0.12.5以降

W&B SDKバージョン0.12.5以降では、W&Bサービスを有効にするためにPythonスクリプトを変更します。`wandb.require` メソッドを使用して、メイン関数内で文字列 `"service"` を渡します：

```python
if __name__ == "__main__":
    main()


def main():
    wandb.require("service")
    # rest-of-your-script-goes-here
```

最適な体験を得るために、最新バージョンにアップグレードすることをお勧めします。

**W&B SDK 0.12.4以前**

W&B SDKバージョン0.12.4以前を使用している場合は、マルチスレッディングを代わりに使用するために`WANDB_START_METHOD`環境変数を `"thread"` に設定します。

### マルチプロセッシングのユースケース例

次のコードスニペットは、高度な分散ユースケースのための一般的な方法を示しています。

#### スポーンプロセス

スポーンされたプロセスでW&B Runを開始する場合は、メイン関数で`wandb.setup()[line 8]` メソッドを使用します：

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

#### W&B Runを共有する

W&B Runオブジェクトを引数として渡すことで、プロセス間でW&B Runsを共有します：

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

:::info
ログの順序について保証できないことに注意してください。同期はスクリプトの作成者に委ねられます。
:::

