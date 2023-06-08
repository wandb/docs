---
description: Use W&B to log distributed training experiments with multiple GPUs.
displayed_sidebar: ja
---

# 分散トレーニング実験の記録

<head>
  <title>分散トレーニング実験の記録</title>
</head>

分散トレーニングでは、複数のGPUを並行して使用してモデルをトレーニングします。W&Bは、分散トレーニング実験をトラッキングするために2つのパターンをサポートしています。

1. **単一プロセス**： W&Bの初期化（[`wandb.init`](https://docs.wandb.ai/ref/python/init)）と実験の記録（[`wandb.log`](https://docs.wandb.ai/ref/python/log)）は単一プロセスで行います。これは、[PyTorch Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP)クラスを使用して分散トレーニング実験のログを記録する一般的な方法です。場合によっては、ユーザーがマルチプロセッシングキューや別の通信プリミティブを使用して、他のプロセスからメインログプロセスにデータを送り込みます。
2. **複数のプロセス**： W&B の初期化（[`wandb.init`](https://docs.wandb.ai/ref/python/init)）と実験ログ（[`wandb.log`](https://docs.wandb.ai/ref/python/log)）は各プロセスごとに行います。各プロセスは実質的に別々の実験となります。W&Bを初期化する際に`group`パラメータを使い、共有実験を定義し、W&BアプリのUI上でログされた値をグループ化するために、`wandb.init(group='group-name')`を使用します。

以下の例では、1台のマシン上の2つのGPUでPyTorch DDPを使用してW&Bを使いメトリクスをトラッキングする方法を説明しています。[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp\_tutorial.html)（`torch.nn` の`DistributedDataParallel`）は、分散トレーニング用の人気のあるライブラリです。基本的な原則はどの分散トレーニング設定にも適用されますが、実装の詳細は異なる場合があります。

:::info
これらの例の背後にあるコードは、W&BのGitHubのexamplesリポジトリ[こちら](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-ddp)でご覧いただけます。特に、1プロセスと複数プロセスの方法を実装する方法については、[`log-dpp.py`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py)のPythonスクリプトを参照してください。
:::

### 方法1: 単一プロセス

この方法では、rank 0のプロセスのみをトラッキングします。この方法を実装するには、W&Bの初期化（`wandb.init`）や、W&Bランの開始、メトリクスのログ（`wandb.log`）をrank 0プロセスで行います。この方法はシンプルで頑健ですが、他のプロセスからのモデルメトリクス（たとえば、損失値やバッチ内の入力データ）はログされません。ただし、システムのメトリクス（使用状況やメモリなど）は、すべてのプロセスで情報が利用可能であるため、すべてのGPUのメトリクスがログされます。

:::info
**この方法は、単一プロセスから利用可能なメトリクスのみをトラッキングするために使用します。** 典型的な例は、GPU/CPUの利用率、共有検証セットでの挙動、勾配やパラメータ、代表的なデータの例における損失値などです。
:::

サンプルのPythonスクリプト（`log-ddp.py`）では、rankが0であるかどうかを確認しています。これには、まず`torch.distributed.launch`を使い複数プロセスを起動し、次にコマンドライン引数`--local_rank`を使いランクを確認します。ランクが0に設定されている場合、[`train()`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py#L24)関数内で条件付きで`wandb`のログを設定します。Pythonスクリプト内では以下のチェックを使用しています：

```python showLineNumbers
if __name__ == "__main__":
    # 引数を取得
    args = parse_args()

    if args.local_rank == 0:  # メインプロセスでのみ実行
        # wandb runを初期化
        run = wandb.init(
            entity=args.entity,
            project=args.project,
        )
        # DDPでモデルを学習
        train(args, run)
    else:
        train(args)
```

W&BアプリのUIを探索して、単一プロセスからトラッキングされたメトリクスの[例のダッシュボード](https://wandb.ai/ayush-thakur/DDP/runs/1s56u3hc/system)を表示します。ダッシュボードには、両方のGPUでトラックされた温度や利用状況などのシステムメトリクスが表示されます。

![](/images/track/distributed_training_method1.png)

ただし、損失値はエポックとバッチサイズの関数として、単一のGPUからのみログに記録されました。

![](/images/experiments/loss_function_single_gpu.png)

### 方法2：多くのプロセス

この方法では、ジョブの各プロセスをトラックし、`wandb.init()`および`wandb.log()`をそれぞれのプロセスから個別に呼び出します。すべてのプロセスが適切に終了するように、トレーニングが終了したら`wandb.finish()`を呼び出すことをお勧めします。

この方法では、ログに記録するための情報がより多くアクセス可能になります。ただし、W&BアプリのUIに複数のW&B Runsが表示されることに注意してください。複数の実験でW&B Runsを追跡するのが難しい場合があります。これを軽減するために、W&Bを初期化するときにgroupパラメータに値を提供して、どのW&B Runが特定の実験に属しているかを追跡します。実験のトレーニングと評価のW&B Runsを追跡する方法の詳細については、[Group Runs](../../runs/grouping.md)を参照してください。
:::info
**個々のプロセスからのメトリクスをトラッキングしたい場合は、このメソッドを使用してください**。典型的な例としては、各ノードのデータや予測（データ配布のデバッグ用）や、メインノード外の個々のバッチのメトリックスがあります。この方法は、すべてのノードからのシステムメトリクスを取得するためにも、メインノードで利用可能なサマリースタティスティクスを取得するためにも必要ありません。
:::

以下のPythonコードスニペットは、W&Bを初期化する際にgroupパラメータを設定する方法を示しています。

```python
if __name__ == "__main__":
    # Get args
    args = parse_args()
    # Initialize run
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group="DDP",  # 実験のすべてのrunを1つのグループにまとめる
    )
    # DDPでモデルをトレーニングする
    train(args, run)
```

W&BアプリのUIを調べて、複数のプロセスからトラッキングされたメトリクスの[例となるダッシュボード](https://wandb.ai/ayush-thakur/DDP?workspace=user-noahluna)を見てください。左サイドバーには、2つのW&BRunsがグループ化されていることに注意してください。グループをクリックして、実験専用のグループページを表示します。専用のグループページには、各プロセスのメトリックスが別々に表示されます。

![](/images/experiments/dashboard_grouped_runs.png)

上の画像は、W&BアプリのUIダッシュボードを示しています。サイドバーには、'null'とラベル付けされた実験と、黄色い枠で囲まれた2つ目の実験が表示されています。グループを展開すると（グループのドロップダウンを選択すると）、その実験に関連するW&BRunsが表示されます。

### よくある分散トレーニングの問題を回避するために、W&Bサービスを使いましょう。

W&Bと分散トレーニングを使っているときに遭遇する可能性がある共通の問題が2つあります。
1. **トレーニングの始めに停滞する** - `wandb`プロセスは、`wandb`のマルチプロセッシングが分散トレーニングのマルチプロセッシングと干渉すると停滞することがあります。
2. **トレーニングの終わりに停滞する** - トレーニングジョブは、`wandb`プロセスが終了する必要があるタイミングを知らない場合に停滞することがあります。Pythonスクリプトの最後で`wandb.finish()`APIを呼び出して、W&BにRunが終了したことを伝えます。wandb.finish() APIはデータのアップロードを完了させ、W&Bを終了させます。

分散ジョブの信頼性を向上させるために、`wandb service`の使用をお勧めします。先述のトレーニングの問題は、wandb serviceが利用できないW&B SDKのバージョンで一般的に見られます。

### W&Bサービスを有効にする

W&B SDKのバージョンによっては、W&Bサービスがデフォルトで有効になっている場合があります。

#### W&B SDK 0.13.0以降

W&B SDK `0.13.0`以降のバージョンでは、W&Bサービスがデフォルトで有効になっています。

#### W&B SDK 0.12.5以降

W&B SDKバージョン0.12.5以降のW&Bサービスを有効にするには、Pythonスクリプトを変更します。`wandb.require`メソッドを使用し、メイン関数内で文字列`"service"`を渡します。

```python
if __name__ == "__main__":
    main()

def main():
    wandb.require("service")
    # ここに残りのスクリプトを記述
```

最適な体験のために、最新バージョンにアップグレードすることをお勧めします。

**W&B SDK 0.12.4以前**
`WANDB_START_METHOD`環境変数を`"thread"`に設定することで、W&B SDKバージョン0.12.4以下を使用している場合にマルチスレッディングを利用できます。

### マルチプロセッシングの例

以下のコードスニペットでは、高度な分散ユースケースのための一般的な方法が示されています。

#### プロセスの生成

`wandb.setup()[line 8]`メソッドをメイン関数内で使用し、生成されたプロセス内でW&B Runを開始します。

```python showLineNumbers
import multiprocessing as mp

def do_work(n):
    run = wandb.init(config=dict(n=n))
    run.log(dict(this=n*n))

def main():
    wandb.setup()
    pool = mp.Pool(processes=4)
    pool.map(do_work, range(4))

if __name__ == "__main__":
    main()
```

#### W&B Runを共有する

W&B Runオブジェクトを引数として渡すことで、プロセス間でW&B Runsを共有できます。
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

ログの順序を保証できないことに注意してください。同期はスクリプトの作者が行うべきです。

:::