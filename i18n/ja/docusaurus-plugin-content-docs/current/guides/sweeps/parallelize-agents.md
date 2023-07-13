---
description: Parallelize W&B Sweep agents on multi-core or multi-GPU machine.
displayed_sidebar: ja
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# エージェントを並列化する

<head>
  <title>エージェントを並列化する</title>
</head>

マルチコアやマルチGPUマシンで W&B スイープエージェントを並列化します。始める前に、W&B スイープを初期化しておくことを確認してください。W&B スイープの初期化方法について詳しくは、[Initialize sweeps](https://docs.wandb.ai/guides/sweeps/initialize-sweeps) を参照してください。

### マルチCPUマシンでの並列化

ユースケースに応じて、次のタブを参照して、CLIを使ったW＆Bスイープエージェントの並列化方法や、Jupyter Notebookでの方法を学習してください。


<Tabs
  defaultValue="cli_text"
  values={[
    {label: 'CLI', value: 'cli_text'},
    {label: 'Jupyter Notebook', value: 'jupyter'},
  ]}>
  <TabItem value="cli_text">

ターミナルを使って、[`wandb agent`](../../ref/cli/wandb-agent.md) コマンドで W&B スイープエージェントを複数のCPUに並列化します。[スイープを初期化](./initialize-sweeps.md)する際に返されたスイープIDを指定してください。

1. ローカルマシン上で複数のターミナルウィンドウを開きます。
2. 下のコードスニペットをコピー＆ペーストし、`sweep_id` をあなたのスイープIDに置き換えます:
```bash
wandb agent sweep_id
```

  </TabItem>
  <TabItem value="jupyter">

Weights & BiasesのPython SDKライブラリを使用して、Jupyter Notebooks内でW&Bスイープエージェントを複数のCPUに並列化します。 [スイープの初期化](./initialize-sweeps.md) 時に返されたsweep IDを確認してください。さらに、 `function` パラメータにスイープが実行する関数名を指定します：

1. 複数のJupyter Notebookを開きます。
2. W&BスイープIDを複数のJupyterノートブックにコピーして貼り付けて、W&Bスイープを並列化します。 たとえば、sweep IDを `sweep_id` という変数に格納し、関数名が `function_name` である場合、次のコードスニペットを複数のjupyterノートブックに貼り付けてスイープを並列化できます：

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```

  </TabItem>
</Tabs>

### 複数のGPUを搭載したマシンでの並列化

CUDA Toolkitを使用して、ターミナルで複数のGPUにW&Bスイープエージェントを並列化する手順に従います。

1. ローカルマシンで複数のターミナルウィンドウを開きます。
2. W&Bスイープジョブ（ [`wandb agent`](https://docs.wandb.ai/ref/cli/wandb-agent)）を開始するときに、 `CUDA_VISIBLE_DEVICES`で使用するGPUインスタンスを指定します。 `CUDA_VISIBLE_DEVICES` に使用するGPUインスタンスに対応する整数値を割り当てます。

たとえば、ローカルマシンに2つのNVIDIA GPUがある場合、ターミナルウィンドウを開いて `CUDA_VISIBLE_DEVICES`を `0` に設定します（ `CUDA_VISIBLE_DEVICES=0`）。次の例の`sweep_ID`を、W&Bスイープを初期化したときに返されたW&BスイープIDに置き換えます:

ターミナル1

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
```

もう一つのターミナルウィンドウを開きます。`CUDA_VISIBLE_DEVICES`を`1`に設定します（`CUDA_VISIBLE_DEVICES=1`）。手前のコードスニペットで言及された`sweep_ID`に同じW&BスイープIDを貼り付けます。

ターミナル2
```bash
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID
```