---
description: Start or stop a W&B Sweep Agent on one or more machines.
displayed_sidebar: ja
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# スイープエージェントの開始

<head>
  <title>W&Bスイープの開始または停止</title>
</head>

1台以上のマシンで1つ以上のエージェントを使用してW&B スイープを開始します。W&Bスイープエージェントは、W&Bスイープ（`wandb sweep`）を初期化したときに起動したWeights & Biasesサーバーにクエリを送信し、それらのハイパーパラメーターを使用してモデルトレーニングを実行します。

W&Bスイープエージェントを開始するには、W&Bスイープを初期化したときに返されたW&BスイープIDを提供してください。W&B スイープIDは以下の形式です：

```bash
entity/project/sweep_ID
```

ここで：

* entity: あなたのWeights & Biases ユーザー名またはチーム名。
* project:  W&B Runの出力を保存するプロジェクトの名前。プロジェクトが指定されていない場合、ランは "Uncategorized" プロジェクトに入れられます。
* sweep\_ID: W&Bによって生成された疑似ランダムでユニークなID。

Jupyter NotebookまたはPythonスクリプト内でW&Bスイープエージェントを開始する場合、W&Bスイープが実行する関数の名前を提供してください。

以下のコードスニペットでは、Weights & Biasesを使用してエージェントを開始する方法を示しています。すでに構成ファイルがあり、W&Bスイープを初期化済みであることを前提としています。構成ファイルの定義方法についての詳細は、[スイープ構成の定義](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)を参照してください。

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'PythonスクリプトまたはJupyterノートブック', value: 'python'},
  ]}>
  <TabItem value="cli">

`sweep_id`に初期化したスイープから返されたスイープIDを指定して、`wandb agent`コマンドを使用してスイープを開始します。下のコードスニペットをコピーして貼り付け、`sweep_id`をあなたのスイープIDに置き換えてください：

```bash
wandb agent sweep_id
```
  </TabItem>
  <TabItem value="python">

Weights & Biases Python SDKライブラリを使用して、スイープを開始します。スイープを初期化したときに返されたスイープIDを提供してください。さらに、スイープが実行する関数の名前を提供してください。

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```
  </TabItem>
</Tabs>

### W&Bエージェントの停止

:::caution
ランダム検索とベイズ検索は無限に実行されます。コマンドライン、Pythonスクリプト内、または[Sweeps UI](./visualize-sweep-results.md)からプロセスを停止する必要があります。
:::

スイープエージェントが試行するW&B Runsの数をオプションで指定できます。以下のコードスニペットは、CLIとJupyterノートブック、Pythonスクリプト内で最大の[W&B Runs](../../ref/python/run.md)を設定する方法を示しています。

<Tabs
  defaultValue="python"
  values={[
    {label: 'PythonスクリプトまたはJupyterノートブック', value: 'python'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="python">

まず、スイープを初期化します。詳細については、[スイープの初期化](https://docs.wandb.ai/guides/sweeps/initialize-sweeps)を参照してください。

```
sweep_id = wandb.sweep(sweep_config)
```

次に、スイープジョブを開始します。スイープ初期化から生成されたスイープIDを提供してください。countパラメータに整数値を渡して、試行する最大run数を設定します。

```python
sweep_id, count = "dtzl1o7u", 10
wandb.agent(sweep_id, count=count)
```
  </TabItem>
  <TabItem value="cli">

まず、[`wandb sweep`](https://docs.wandb.ai/ref/cli/wandb-sweep)コマンドを使ってスイープを初期化します。詳細については、[スイープの初期化](https://docs.wandb.ai/guides/sweeps/initialize-sweeps)を参照してください。

```
wandb sweep config.yaml
```

countフラグに整数値を渡して、試行する最大run数を設定します。

```
NUM=10
SWEEPID="dtzl1o7u"
wandb agent --count $NUM $SWEEPID
```
  </TabItem>
</Tabs>

###