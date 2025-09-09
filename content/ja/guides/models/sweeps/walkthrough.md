---
title: 'チュートリアル: sweep を定義・初期化・実行する'
description: Sweeps クイックスタート は、sweep を定義し、初期化し、実行する方法を説明します。主な手順は 4 つあります。
menu:
  default:
    identifier: ja-guides-models-sweeps-walkthrough
    parent: sweeps
weight: 1
---

このページでは、sweep を定義・初期化して実行する方法を説明します。主なステップは 4 つあります:

1. [トレーニング用のコードを準備する]({{< relref path="#set-up-your-training-code" lang="ja" >}})
2. [sweep configuration で探索空間を定義する]({{< relref path="#define-the-search-space-with-a-sweep-configuration" lang="ja" >}})
3. [sweep を初期化する]({{< relref path="#initialize-the-sweep" lang="ja" >}})
4. [sweep agent を開始する]({{< relref path="#start-the-sweep" lang="ja" >}})

次の コード を Jupyter Notebook または Python スクリプトにコピー & ペーストしてください:

```python
# W&B の Python ライブラリをインポートし、W&B にログイン
import wandb

# 1: 目的/トレーニング関数を定義
def objective(config):
    score = config.x**3 + config.y
    return score

def main():
    with wandb.init(project="my-first-sweep") as run:
        score = objective(run.config)
        run.log({"score": score})

# 2: 探索空間を定義
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

# 3: sweep を開始
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```

以下のセクションで、サンプル コード の各ステップを解説します。

## トレーニング用のコードを準備する
`wandb.Run.config` から ハイパーパラメーター の 値 を受け取り、それを使って モデル を学習し、メトリクス を返す トレーニング関数 を定義します。

必要に応じて、W&B の Run の出力を保存したい project 名を指定します（[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) の project パラメータ）。project を指定しない場合、run は "Uncategorized" project に配置されます。

{{% alert %}}
sweep と run は同じ project に属している必要があります。したがって、W&B を初期化するときに指定する名前は、sweep を初期化するときに指定する project 名と一致していなければなりません。
{{% /alert %}}

```python
# 1: 目的/トレーニング関数を定義
def objective(config):
    score = config.x**3 + config.y
    return score


def main():
    with wandb.init(project="my-first-sweep") as run:
        score = objective(run.config)
        run.log({"score": score})
```

## sweep configuration で探索空間を定義する

スイープ対象の ハイパーパラメーター を 辞書 に指定します。利用可能な 設定 オプションは、[Define sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。

次の例は、ランダム検索（`'method':'random'`）を使う sweep configuration を示しています。sweep は、設定に記載された バッチサイズ、エポック、学習率 の 値 からランダムに組み合わせを選びます。

`"goal": "minimize"` が指定されている場合、W&B は `metric` キーで指定されたメトリクスを最小化します。この例では、`"name": "score"` により、メトリクス `score` の最小化が最適化目標になります。

```python
# 2: 探索空間を定義
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}
```

## Sweep を初期化する

W&B は _Sweep Controller_ を使って、クラウド（standard）や 1 台以上のマシンにまたがるローカル（local）で sweeps を管理します。_Sweep Controller_ について詳しくは、[Search and stop algorithms locally]({{< relref path="./local-controller.md" lang="ja" >}}) を参照してください。

sweep を初期化すると、識別番号が返されます:

```python
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
```

sweep の初期化について詳しくは、[Initialize sweeps]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

## Sweep を開始する

sweep を開始するには、[`wandb.agent()`]({{< relref path="/ref/python/sdk/functions/agent.md" lang="ja" >}}) API を呼び出します。

```python
wandb.agent(sweep_id, function=main, count=10)
```

{{% alert color="secondary" title="マルチプロセッシング" %}}
Python 標準ライブラリの `multiprocessing` や PyTorch の `pytorch.multiprocessing` パッケージを使う場合は、`wandb.agent()` と `wandb.sweep()` の呼び出しを `if __name__ == '__main__':` で囲む必要があります。例えば次のようにします:

```python
if __name__ == '__main__':
    wandb.agent(sweep_id="<sweep_id>", function="<function>", count="<count>")
```

この慣習で コード をラップすると、スクリプトが直接実行されたときにのみ実行され、ワーカープロセスでモジュールとしてインポートされたときには実行されないことが保証されます。

マルチプロセッシングの詳細は、[Python standard library `multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods) や [PyTorch `multiprocessing`](https://pytorch.org/docs/stable/notes/multiprocessing.html#asynchronous-multiprocess-training-e-g-hogwild) を参照してください。`if __name__ == '__main__':` の慣習については https://realpython.com/if-name-main-python/ を参照してください。
{{% /alert %}}

## 結果を可視化する (任意)

project を開くと、W&B App の ダッシュボード でライブの 結果 を確認できます。数クリックで、[parallel coordinates plots]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}})、[parameter importance の分析]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}})、[他のグラフ種類]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) など、豊富で インタラクティブな グラフ を作成できます。

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Sweeps Dashboard の例" >}}

可視化の詳しい方法は、[Visualize sweep results]({{< relref path="./visualize-sweep-results.md" lang="ja" >}}) を参照してください。ダッシュボードの例は、このサンプル [Sweeps Project](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3) をご覧ください。

## エージェントを停止する (任意)

ターミナルで `Ctrl+C` を押すと現在の run が停止します。もう一度押すと エージェント を終了します。