---
title: エージェントを並列化する
description: マルチコアやマルチ GPU マシンで W&B sweep agent を並列実行する。
menu:
  default:
    identifier: ja-guides-models-sweeps-parallelize-agents
    parent: sweeps
weight: 6
---

W&B Sweep エージェントをマルチコアまたはマルチ GPU マシン上で並列化できます。始める前に、W&B Sweep を初期化済みであることを確認してください。W&B Sweep の初期化方法について詳しくは、[スイープの初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}})をご参照ください。

### マルチ CPU マシンでの並列化

ユースケースに応じて、以下のタブで CLI や Jupyter Notebook を使った W&B Sweep エージェントの並列化方法を学ぶことができます。

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
[`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}}) コマンドを使用して、ターミナル上で sweep agent を複数の CPU に並列化できます。[スイープの初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}})時に返された sweep ID を指定してください。

1. ローカルマシンで 2 つ以上のターミナルウィンドウを開きます。
2. 下記のコードスニペットをコピー＆ペーストし、`sweep_id` を自分の sweep ID に置き換えます。

```bash
wandb agent sweep_id
```
  {{% /tab %}}
  {{% tab header="Jupyter Notebook" %}}
W&B Python SDK ライブラリを用いて、Jupyter Notebook 上で複数の CPU で W&B Sweep agent を並列実行できます。[スイープの初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}})時に返された sweep ID を用意してください。さらに、sweep で実行する関数名を `function` パラメータとして渡します。

1. 複数の Jupyter Notebook を開きます。
2. W&B Sweep ID を複数の Jupyter Notebook に貼り付けることで、W&B Sweep を並列化できます。例えば、Sweep ID が `sweep_id` という変数に格納されており、実行したい関数名が `function_name` の場合、以下のコードスニペットを複数のノートブックで実行できます。

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```
  {{% /tab %}}
{{< /tabpane >}}

### マルチ GPU マシンでの並列化

CUDA Toolkit を使って、ターミナル上で W&B Sweep agent を複数の GPU で並列化する手順は次の通りです。

1. ローカルマシンで 2 つ以上のターミナルウィンドウを開きます。
2. W&B Sweep ジョブ（[`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}})）を開始する際、`CUDA_VISIBLE_DEVICES` で利用する GPU インスタンスを指定します。`CUDA_VISIBLE_DEVICES` には利用したい GPU 番号（整数値）を指定してください。

例えば、ローカルマシン上に NVIDIA GPU が 2 枚ある場合、1 つめのターミナルウィンドウで `CUDA_VISIBLE_DEVICES` を `0` （`CUDA_VISIBLE_DEVICES=0`）に設定して実行します。以下の例で `sweep_ID` は、W&B Sweep の初期化時に取得した Sweep ID に置き換えてください。

ターミナル 1

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
```

2 つめのターミナルウィンドウを開き、`CUDA_VISIBLE_DEVICES` を `1` （`CUDA_VISIBLE_DEVICES=1`）に設定します。同じ `sweep_ID` を指定して、以下のように実行してください。

ターミナル 2

```bash
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID
```