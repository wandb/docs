---
title: Parallelize agents
description: マルチコアまたはマルチ GPU マシンで W&B Sweep エージェント を並列化します。
menu:
  default:
    identifier: ja-guides-models-sweeps-parallelize-agents
    parent: sweeps
weight: 6
---

マルチコアまたはマルチ GPU マシンで W&B Sweep エージェントを並列化します。開始する前に、W&B Sweep を初期化していることを確認してください。W&B Sweep の初期化方法の詳細については、[Sweeps の初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

### マルチ CPU マシンでの並列化

ユースケースに応じて、次のタブで CLI を使用するか、Jupyter Notebook 内で W&B Sweep エージェントを並列化する方法を検討してください。

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
[`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}}) コマンドを使用して、ターミナルで複数の CPU に W&B Sweep エージェントを並列化します。[sweep を初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) したときに返された sweep ID を指定します。

1. ローカルマシンで複数のターミナルウィンドウを開きます。
2. 次の コードスニペット をコピーして貼り付け、`sweep_id` を sweep ID に置き換えます。

```bash
wandb agent sweep_id
```  
  {{% /tab %}}
  {{% tab header="Jupyter Notebook" %}}
W&B Python SDK ライブラリを使用して、Jupyter Notebook 内で複数の CPU に W&B Sweep エージェントを並列化します。[sweep を初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) したときに返された sweep ID があることを確認してください。さらに、sweep が `function` パラメータに対して実行する関数の名前を指定します。

1. 複数の Jupyter Notebook を開きます。
2. 複数の Jupyter Notebook に W&B Sweep ID をコピーして貼り付け、W&B Sweep を並列化します。たとえば、sweep ID が `sweep_id` という変数に格納され、関数の名前が `function_name` の場合、次の コードスニペット を複数の Jupyter Notebook に貼り付けて、sweep を並列化できます。

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```  
  {{% /tab %}}
{{< /tabpane >}}

### マルチ GPU マシンでの並列化

次の手順に従って、CUDA Toolkit を使用してターミナルで複数の GPU に W&B Sweep エージェントを並列化します。

1. ローカルマシンで複数のターミナルウィンドウを開きます。
2. W&B Sweep ジョブ ([`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}})) を開始するときに、`CUDA_VISIBLE_DEVICES` で使用する GPU インスタンスを指定します。`CUDA_VISIBLE_DEVICES` に、使用する GPU インスタンスに対応する整数値を割り当てます。

たとえば、ローカルマシンに 2 つの NVIDIA GPU があるとします。ターミナルウィンドウを開き、`CUDA_VISIBLE_DEVICES` を `0` (`CUDA_VISIBLE_DEVICES=0`) に設定します。次の例の `sweep_ID` を、W&B Sweep を初期化したときに返される W&B Sweep ID に置き換えます。

ターミナル 1

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
```

2 番目のターミナルウィンドウを開きます。`CUDA_VISIBLE_DEVICES` を `1` (`CUDA_VISIBLE_DEVICES=1`) に設定します。上記の コードスニペット で説明されている `sweep_ID` に同じ W&B Sweep ID を貼り付けます。

ターミナル 2

```bash
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID
```