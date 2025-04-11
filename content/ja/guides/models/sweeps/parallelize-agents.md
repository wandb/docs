---
title: エージェントの並列化
description: マルチコアまたはマルチGPUマシンでW&B sweep agentを並列化します。
menu:
  default:
    identifier: ja-guides-models-sweeps-parallelize-agents
    parent: sweeps
weight: 6
---

W&B スイープエージェントをマルチコアまたはマルチ GPU マシンで並列化しましょう。始める前に、W&B スイープが初期化されていることを確認してください。W&B スイープの初期化方法についての詳細は、[Initialize sweeps]({{< relref path="./initialize-sweeps.md" lang="ja" >}})をご覧ください。

### マルチ CPU マシンで並列化

ユースケースに応じて、以下のタブを参照し、CLI や Jupyter ノートブック内で W&B スイープエージェントを並列化する方法を学びましょう。

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
[`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}}) コマンドを使用して、ターミナルで W&B スイープエージェントを複数の CPU に渡って並列化します。[sweep を初期化したとき]({{< relref path="./initialize-sweeps.md" lang="ja" >}})に返されたスイープ ID を提供してください。

1. ローカルマシンで複数のターミナルウィンドウを開きます。
2. 以下のコードスニペットをコピーして貼り付け、`sweep_id` をあなたのスイープ ID に置き換えます:

```bash
wandb agent sweep_id
```  
  {{% /tab %}}
  {{% tab header="Jupyter Notebook" %}}
W&B Python SDK ライブラリを使用して、Jupyter ノートブック内で W&B スイープエージェントを複数の CPU に渡って並列化します。[sweep を初期化したとき]({{< relref path="./initialize-sweeps.md" lang="ja" >}})に返されたスイープ ID を確認してください。さらに、スイープが実行する関数の名前を `function` パラメータに提供します。

1. 複数の Jupyter ノートブックを開きます。
2. 複数の Jupyter ノートブックに W&B スイープ ID をコピーして貼り付け、W&B スイープを並列化します。例えば、`sweep_id` という変数にスイープ ID が保存されていて、関数の名前が `function_name` である場合、以下のコードスニペットを複数の Jupyter ノートブックに貼り付けることができます:

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```  
  {{% /tab %}}
{{< /tabpane >}}

### マルチ GPU マシンで並列化

CUDA Toolkit を使用して、ターミナルで W&B スイープエージェントを複数の GPU に渡って並列化するための手順に従ってください。

1. ローカルマシンで複数のターミナルウィンドウを開きます。
2. W&B スイープジョブを開始するときに `CUDA_VISIBLE_DEVICES` を使用して使用する GPU インスタンスを指定します（[`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}})）。`CUDA_VISIBLE_DEVICES` に使用する GPU インスタンスに対応する整数値を割り当てます。

例えば、ローカルマシンに 2 つの NVIDIA GPU があると仮定します。ターミナルウィンドウを開き、`CUDA_VISIBLE_DEVICES` を `0`（`CUDA_VISIBLE_DEVICES=0`）に設定します。以下の例で、`sweep_ID` を初期化したときに返された W&B スイープ ID に置き換えます:

ターミナル 1

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
```

2 番目のターミナルウィンドウを開きます。`CUDA_VISIBLE_DEVICES` を `1`（`CUDA_VISIBLE_DEVICES=1`）に設定します。次のコードスニペットで言及された `sweep_ID` に同じ W&B スイープ ID を貼り付けます:

ターミナル 2

```bash
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID
```