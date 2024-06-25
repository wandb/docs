---
description: マルチコアまたはマルチGPUマシン上でW&B sweep エージェントを並列処理する。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Parallelize agents

<head>
  <title>Parallelize agents</title>
</head>

W&B Sweep エージェントをマルチコアまたはマルチGPUマシンで並列化します。始める前に、W&B Sweep を初期化していることを確認してください。W&B Sweep の初期化方法については、[Initialize sweeps](./initialize-sweeps.md)を参照してください。

### マルチCPUマシンで並列化する

ユースケースに応じて、以下のタブを参照し、CLI または Jupyter Notebook を使用して W&B Sweep エージェントを並列化する方法を学びましょう。

<Tabs
  defaultValue="cli_text"
  values={[
    {label: 'CLI', value: 'cli_text'},
    {label: 'Jupyter Notebook', value: 'jupyter'},
  ]}>
  <TabItem value="cli_text">

[`wandb agent`](../../ref/cli/wandb-agent.md) コマンドを使用して、ターミナルで W&B Sweep エージェントを複数のCPUに並列化します。sweep を [初期化](./initialize-sweeps.md)したときに返される sweep ID を提供します。

1. ローカルマシンで複数のターミナルウィンドウを開きます。
2. 以下のコードスニペットをコピーして貼り付け、`sweep_id` をあなたの sweep ID に置き換えます:

```bash
wandb agent sweep_id
```

  </TabItem>
  <TabItem value="jupyter">

W&B Python SDK ライブラリを使用して、Jupyter Notebook 内で W&B Sweep エージェントを複数のCPUに並列化します。sweep を [初期化](./initialize-sweeps.md)したときに返される sweep ID を持っていることを確認してください。さらに、sweep が実行する関数の名前を `function` パラメータに提供します:

1. 複数の Jupyter Notebook を開きます。
2. 複数の Jupyter Notebook に W&B Sweep ID をコピーして、W&B Sweep を並列化します。例えば、`sweep_id` という変数に sweep ID が格納され、関数名が `function_name` の場合、以下のコードスニペットを複数の Jupyter Notebook に貼り付けて、sweep を並列化できます:

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```

  </TabItem>
</Tabs>

### マルチGPUマシンで並列化する

以下の手順に従って、ターミナルで CUDA Toolkit を使用して W&B Sweep エージェントを複数のGPUに並列化します:

1. ローカルマシンで複数のターミナルウィンドウを開きます。
2. W&B Sweep ジョブを開始するときに `CUDA_VISIBLE_DEVICES` を指定して使用する GPU インスタンスを設定します ([`wandb agent`](../../ref/cli/wandb-agent.md))。`CUDA_VISIBLE_DEVICES` に使用する GPU インスタンスに対応する整数値を割り当てます。

例えば、ローカルマシンに2つのNVIDIA GPUがあるとします。ターミナルウィンドウを開き、`CUDA_VISIBLE_DEVICES` を `0` に設定します (`CUDA_VISIBLE_DEVICES=0`)。以下の例で `sweep_ID` を、sweep を初期化したときに返される W&B Sweep ID に置き換えます:

ターミナル 1

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
```

2つ目のターミナルウィンドウを開きます。`CUDA_VISIBLE_DEVICES` を `1` に設定します (`CUDA_VISIBLE_DEVICES=1`)。以下のコードスニペットで言及された同じ W&B Sweep ID を `sweep_ID` に貼り付けます:

ターミナル 2

```bash
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID
```