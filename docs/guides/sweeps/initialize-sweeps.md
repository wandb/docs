---
description: W&B Sweep を初期化する
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Initialize sweeps

<head>
  <title>Start a W&B Sweep</title>
</head>

W&B は _Sweep Controller_ を使用してクラウド（標準）またはローカル（ローカル）で複数のマシンにわたる sweeps を管理します。run が完了すると、sweep controller は新しい run を実行するための新しい指示セットを発行します。これらの指示は、実際に run を実行する _agents_ に拾われます。通常の W&B Sweep では、controller は W&B サーバー上にあり、agents は _あなた_ のマシン上に存在します。

以下のコードスニペットは、CLI および Jupyter Notebook や Python スクリプト内で sweeps を初期化する方法を示しています。

:::caution
1. Sweep を初期化する前に、YAML ファイルまたはスクリプト内のネストされた Python 辞書オブジェクトで定義された sweep configuration があることを確認してください。詳細については、[Define sweep configuration](../../guides/sweeps/define-sweep-configuration.md) を参照してください。
2. W&B Sweep と W&B Run は同じプロジェクト内になければなりません。したがって、W&B を初期化するときに提供する名前 ([`wandb.init`](../../ref/python/init.md)) は、W&B Sweep を初期化するときに提供するプロジェクトの名前 ([`wandb.sweep`](../../ref/python/sweep.md)) と一致する必要があります。
:::

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python script or Jupyter Notebook', value: 'python'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="python">

W&B SDK を使用して sweep を初期化します。sweep configuration 辞書を `sweep` パラメータに渡します。また、W&B Run の出力を保存したいプロジェクトの名前を project パラメータ (`project`) にオプションで提供できます。プロジェクトが指定されていない場合、run は「Uncategorized」というプロジェクトに入れられます。

```python
import wandb

# 実例 sweep configuration
sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [5, 10, 15]},
        "lr": {"max": 0.1, "min": 0.0001},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="project-name")
```

[`wandb.sweep`](../../ref/python/sweep) 関数は sweep ID を返します。sweep ID にはエンティティ名とプロジェクト名が含まれます。sweep ID をメモしておいてください。
  </TabItem>
  <TabItem value="cli">

W&B CLI を使用して sweep を初期化します。設定ファイルの名前を指定してください。オプションで、`project` フラグにプロジェクトの名前を指定できます。プロジェクトが指定されていない場合、W&B Run は「Uncategorized」というプロジェクトに入れられます。

[`wandb sweep`](../../ref/cli/wandb-sweep) コマンドを使用して sweep を初期化します。次のコード例では、`sweeps_demo` プロジェクトの sweep を初期化し、`config.yaml` ファイルを設定に使用します。

```bash
wandb sweep --project sweeps_demo config.yaml
```

このコマンドは sweep ID を出力します。sweep ID にはエンティティ名とプロジェクト名が含まれます。sweep ID をメモしておいてください。
  </TabItem>
</Tabs>