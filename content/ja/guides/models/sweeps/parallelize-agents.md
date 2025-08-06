---
title: エージェントを並列化する
description: 複数のコアや複数の GPU を搭載したマシンで、W&B sweep agent を並列実行する。
menu:
  default:
    identifier: parallelize-agents
    parent: sweeps
weight: 6
---

W&B Sweep エージェントをマルチコアやマルチ GPU マシンで並列化しましょう。始める前に、W&B Sweep が初期化されていることを確認してください。W&B Sweep の初期化方法については、[Initialize sweeps]({{< relref "./initialize-sweeps.md" >}}) をご覧ください。

### マルチ CPU マシンでの並列化

ユースケースに応じて、以下のタブを切り替えて CLI や Jupyter Notebook で W&B Sweep エージェントを並列化する方法を学べます。

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
[`wandb agent`]({{< relref "/ref/cli/wandb-agent.md" >}}) コマンドを使い、ターミナルから複数の CPU に Sweep エージェントを並列実行できます。Sweep を [初期化したとき]({{< relref "./initialize-sweeps.md" >}}) に得られた Sweep ID を指定しましょう。

1. ローカルマシンでターミナルウィンドウを複数開きます。
2. 下記のコードスニペットをコピーして貼り付け、`sweep_id` を自分の Sweep ID に置き換えます。

```bash
wandb agent sweep_id
```  
  {{% /tab %}}
  {{% tab header="Jupyter Notebook" %}}
W&B Python SDK ライブラリを使えば、Jupyter Notebook 内でも複数 CPU を使って W&B Sweep エージェントを並列実行できます。Sweep を [初期化した際]({{< relref "./initialize-sweeps.md" >}}) に返される Sweep ID を用意してください。また、`function` パラメータとしてエージェントが実行する関数名を渡します。

1. 複数の Jupyter Notebook を開きます。
2. 複数の Jupyter Notebook で W&B Sweep ID を貼り付けて並列実行を行います。例えば、Sweep ID を `sweep_id` という変数、実行する関数名を `function_name` として下記のコードスニペットを複数の Notebook に貼り付けてください。

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```  
  {{% /tab %}}
{{< /tabpane >}}



### マルチ GPU マシンでの並列化

CUDA Toolkit を利用して、ターミナルから複数の GPU で W&B Sweep エージェントを並列実行する手順を紹介します。

1. ローカルマシンでターミナルウィンドウを複数開きます。
2. 各 W&B Sweep ジョブ（[`wandb agent`]({{< relref "/ref/cli/wandb-agent.md" >}})）を起動する際、`CUDA_VISIBLE_DEVICES` で使いたい GPU インスタンスを指定します。`CUDA_VISIBLE_DEVICES` には割り当てたい GPU の番号（整数）を指定してください。

例えば、ローカルマシンに NVIDIA GPU が 2 枚搭載されている場合、下記のようにそれぞれ設定します。`sweep_ID` の部分は、Sweep 初期化時に返される W&B Sweep ID に置き換えてください。

ターミナル 1

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
```

2 つ目のターミナルウィンドウを開きます。`CUDA_VISIBLE_DEVICES` を `1` に設定し、同じ `sweep_ID` を使って起動します。

ターミナル 2

```bash
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID
```