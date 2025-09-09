---
title: エージェントを並列化する
description: マルチコアまたはマルチ GPU マシンで W&B Sweep agents を並列実行する。
menu:
  default:
    identifier: ja-guides-models-sweeps-parallelize-agents
    parent: sweeps
weight: 6
---

マルチコアまたはマルチ GPU マシンで W&B Sweep agents を並列化します。始める前に、W&B Sweep を初期化していることを確認してください。W&B Sweep の初期化方法については [Sweeps の初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

### マルチ CPU マシンでの並列化

ユースケースに応じて、以下のタブで CLI または Jupyter Notebook 内で W&B Sweep agents を並列化する方法を確認してください。


{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
[`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}}) コマンドを使い、ターミナルから複数 CPU にまたがって sweep agent を並列化します。[sweep を初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) したときに返された sweep ID を用意してください。 

1. ローカルマシンで複数のターミナルウィンドウを開きます。
2. 下のコードスニペットをコピー＆ペーストし、`sweep_id` を自分の sweep ID に置き換えます:

```bash
wandb agent sweep_id
```  
  {{% /tab %}}
  {{% tab header="Jupyter Notebook" %}}
W&B Python SDK ライブラリを使い、Jupyter Notebook 内で複数 CPU にまたがって W&B Sweep agent を並列化します。[sweep を初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) したときに返された sweep ID を用意してください。加えて、`function` パラメータには sweep が実行する関数名を指定します:

1. 複数の Jupyter Notebook を開きます。
2. 複数の Jupyter Notebook に同じ W&B Sweep ID を貼り付けて W&B Sweep を並列化します。例えば、`sweep_id` という変数に sweep ID が保存され、関数名が `function_name` の場合、次のコードスニペットを複数の Jupyter Notebook に貼り付ければ sweep を並列化できます: 

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```  
  {{% /tab %}}
{{< /tabpane >}}



### マルチ GPU マシンでの並列化

CUDA Toolkit を使い、ターミナルから複数 GPU で W&B Sweep agent を並列化する手順は次のとおりです:

1. ローカルマシンで複数のターミナルウィンドウを開きます。
2. W&B Sweep ジョブ（[`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}})）を開始するときに、使用する GPU インスタンスを `CUDA_VISIBLE_DEVICES` で指定します。`CUDA_VISIBLE_DEVICES` には使用する GPU インスタンスに対応する整数値を設定します。

例えば、ローカルマシンに NVIDIA GPU が 2 基あるとします。ターミナルウィンドウを開き、`CUDA_VISIBLE_DEVICES` を `0`（`CUDA_VISIBLE_DEVICES=0`）に設定します。以下の例の `sweep_ID` は、W&B Sweep を初期化したときに返された W&B Sweep ID に置き換えてください:

ターミナル 1

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
```

2 つ目のターミナルウィンドウを開きます。`CUDA_VISIBLE_DEVICES` を `1`（`CUDA_VISIBLE_DEVICES=1`）に設定します。先ほどのコードスニペット中の `sweep_ID` には、同じ W&B Sweep ID を指定します:

ターミナル 2

```bash
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID
```