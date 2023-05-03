---
description: W&Bスイープを初期化する
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# スイープを初期化する

<head>
  <title>W&Bスイープを開始する</title>
</head>

Weights & Biasesは、_Sweep Controller_ を使用して、クラウド（標準）、ローカル（local）の1台以上のマシンでスイープを管理します。1つのrunが完了すると、スイープコントローラは、実行する新しいrunを説明する新しい指示セットを発行します。これらの指示は、実際にrunを実行する_agents_によって受け取られます。典型的なW&Bスイープでは、コントローラはWeights & Biasesサーバに存在します。エージェントは_あなたの_マシンに存在します。

以下のコードスニペットでは、CLIおよびJupyterノートブックやPythonスクリプト内でスイープを初期化する方法を示しています。

:::caution
1. スイープを初期化する前に、スイープ構成がYAMLファイルまたはスクリプト内のネストされたPythonディクショナリオブジェクトで定義されていることを確認してください。詳細については、[スイープ構成の定義](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)を参照してください。
2. W&BスイープとW&B Runは同じプロジェクト内にある必要があります。つまり、Weights & Biasesを初期化する際に提供する名前（[`wandb.init`](https://docs.wandb.ai/ref/python/init)）は、W&Bスイープを初期化する際に提供するプロジェクトの名前（[`wandb.sweep`](https://docs.wandb.ai/ref/python/sweep)）と一致する必要があります。
:::

<Tabs
  defaultValue="python"
  values={[
    {label: 'PythonスクリプトまたはJupyterノートブック', value: 'python'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="python">

Weights & Biases SDKを使用してスイープを初期化します。`sweep`パラメータにスイープ構成辞書を渡します。必要に応じて、W&B Runの出力が格納されるプロジェクトの名前を、プロジェクトパラメータ（`project`）に指定します。プロジェクトが指定されていない場合、runは"未分類"プロジェクトに格納されます。
```python
import wandb

# 例: スイープ構成
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize', 
        'name': 'val_acc'
		},
    'parameters': {
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [5, 10, 15]},
        'lr': {'max': 0.1, 'min': 0.0001}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="project-name")
```

[`wandb.sweep`](https://docs.wandb.ai/ref/python/sweep)関数は、スイープIDを返します。スイープIDには、エンティティ名とプロジェクト名が含まれます。スイープIDをメモしておいてください。
  </TabItem>
  <TabItem value="cli">

Weights＆Biases CLIを使用して、スイープを初期化します。設定ファイルの名前を指定してください。また、`project`フラグにプロジェクト名を指定しても構いません。プロジェクトが指定されていない場合、W&B Runは「Uncategorized」プロジェクトに入れられます。

[`wandb sweep`](https://docs.wandb.ai/ref/cli/wandb-sweep)コマンドを使って、スイープを初期化します。次のコード例では、`sweeps_demo`プロジェクトのスイープが初期化され、設定には`config.yaml`ファイルが使用されています。

```bash
wandb sweep --project sweeps_demo config.yaml
```
このコマンドは、スイープIDを出力します。スイープIDには、エンティティ名とプロジェクト名が含まれています。スイープIDをメモしておいてください。

  </TabItem>

</Tabs>