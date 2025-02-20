---
title: Parallelize agents
description: マルチコアまたはマルチ GPU マシンで W&B sweep agent を並列化する。
menu:
  default:
    identifier: ja-guides-models-sweeps-parallelize-agents
    parent: sweeps
weight: 6
---

W&B Sweep エージェントをマルチコアまたはマルチ GPU マシンで並列化します。始める前に、W&B Sweep が初期化されていることを確認してください。W&B Sweep の初期化方法の詳細については、[Initialize sweeps]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

### マルチCPUマシンでの並列化

ユースケースに応じて、次のタブを探索し、CLI または Jupyter Notebook 内で W&B Sweep エージェントを並列化する方法を学びます。

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
[`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}}) コマンドを使用して、W&B Sweep エージェントを複数の CPU にわたってターミナルで並列化します。選択したスイープ ID は、[sweep を初期化した]({{< relref path="./initialize-sweeps.md" lang="ja" >}})際に返されます。

1. ローカルマシンで複数のターミナルウィンドウを開きます。
2. 以下のコードスニペットをコピーアンドペーストし、`sweep_id` をあなたのスイープ ID に置き換えてください:

```bash
wandb agent sweep_id
```  
  {{% /tab %}}
  {{% tab header="Jupyter Notebook" %}}
W&B Python SDK ライブラリを使用して、W&B Sweep エージェントを Jupyter Notebooks 内で複数の CPU に並列化します。選択したスイープ ID は、[sweep を初期化した]({{< relref path="./initialize-sweeps.md" lang="ja" >}})際に返されます。また、sweep が実行する関数の名前を `function` パラメータで指定します:

1. 複数の Jupyter Notebook を開きます。
2. W&B Sweep ID を複数の Jupyter Notebook にコピーして、W&B Sweep を並列化します。たとえば、スイープ ID が `sweep_id` という変数に格納されていて、関数の名前が `function_name` の場合、以下のコードスニペットを複数の Jupyter Notebook に貼り付けてスイープを並列化できます:

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```  
  {{% /tab %}}
{{< /tabpane >}}


### マルチGPUマシンでの並列化

CUDA ツールキットを使用して、W&B Sweep エージェントを複数の GPU にわたってターミナルで並列化する手順に従ってください:

1. ローカルマシンで複数のターミナルウィンドウを開きます。
2. W&B Sweep ジョブを開始するときに `CUDA_VISIBLE_DEVICES` を指定して使用する GPU インスタンスを指定します ([`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}}))。`CUDA_VISIBLE_DEVICES` には使用する GPU インスタンスに対応する整数値を指定します。

たとえば、ローカルマシンに NVIDIA GPU が 2 台あるとします。ターミナルウィンドウを開き、`CUDA_VISIBLE_DEVICES` を `0` (`CUDA_VISIBLE_DEVICES=0`) に設定します。次の例の `sweep_ID` を W&B Sweep を初期化したときに返された W&B Sweep ID に置き換えてください:

ターミナル 1

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
```

2 番目のターミナルウィンドウを開きます。`CUDA_VISIBLE_DEVICES` を `1` (`CUDA_VISIBLE_DEVICES=1`) に設定します。同じ W&B Sweep ID を以下のコードスニペットで示された `sweep_ID` に貼り付けます:

ターミナル 2

```bash
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID
```