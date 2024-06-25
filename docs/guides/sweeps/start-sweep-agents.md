---
description: 1台以上のマシンで W&B Sweep Agent を開始または停止します。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Start sweep agents

<head>
  <title>Start or stop a W&B Sweep</title>
</head>

1つ以上のマシンで1つ以上のエージェントを使用してW&B Sweepを開始します。W&B sweep agentは、W&B Sweepを初期化した際に立ち上げたW&Bサーバーに問い合わせを行い、ハイパーパラメーターを取得してモデルトレーニングを実行します。

W&B sweep agentを開始するには、W&B Sweepを初期化した際に返されたW&B Sweep IDを指定します。W&B Sweep IDの形式は以下の通りです：

```bash
entity/project/sweep_ID
```

以下の項目が含まれます：

* entity: あなたのW&Bのユーザー名またはチーム名。
* project: W&B Runの出力を保存したいプロジェクトの名前。プロジェクトが指定されていない場合、runは「未分類」プロジェクトに配置されます。
* sweep\_ID: W&Bによって生成される擬似乱数のユニークID。

Jupyter NotebookまたはPython script内でW&B sweep agentを開始する場合、W&B Sweepが実行する関数の名前を指定します。

以下のコードスニペットは、W&Bを使用してエージェントを開始する方法を示しています。既に設定ファイルを持ち、W&B Sweepを初期化したものと仮定します。設定ファイルの定義方法については、[Define sweep configuration](./define-sweep-configuration.md)を参照してください。

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'Python script or Jupyter Notebook', value: 'python'},
  ]}>
  <TabItem value="cli">

`sweep_id`を初期化した際に返されたSweep IDに置き換えて、以下のコードスニペットをコピーして貼り付けます。`wandb agent` コマンドを使用してsweepを開始します：

```bash
wandb agent sweep_id
```
  </TabItem>
  <TabItem value="python">

W&B Python SDK libraryを使用してsweepを開始します。初期化した際に返されたsweep IDを指定します。さらに、sweepが実行する関数の名前も指定します。

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```
  </TabItem>
</Tabs>

### Stop W&B agent

:::caution
ランダム探索およびベイズ探索は永遠に実行されます。コマンドライン、Python script内、または[Sweeps UI](./visualize-sweep-results.md)からプロセスを停止する必要があります。
:::

オプションで、Sweep agentが試すべきW&B Runsの数を指定できます。以下のコードスニペットは、CLIおよびJupyter Notebook、Python script内で最大数のW&B Runsを設定する方法を示しています。

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python script or Jupyter Notebook', value: 'python'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="python">

まず、sweepを初期化します。詳細については[Initialize sweeps](./initialize-sweeps.md)を参照してください。

```
sweep_id = wandb.sweep(sweep_config)
```

次に、sweepジョブを開始します。sweep開始時に生成されたsweep IDを指定します。整数値をcountパラメーターに渡して、試行するrunsの最大数を設定します。

```python
sweep_id, count = "dtzl1o7u", 10
wandb.agent(sweep_id, count=count)
```

:::caution
もし、同じscriptやノートブック内でsweep agentが終了した後に新しいrunを開始する場合は、新しいrunを開始する前に `wandb.teardown()` を呼び出す必要があります。
:::


  </TabItem>

  <TabItem value="cli">

まず、[`wandb sweep`](../../ref/cli/wandb-sweep.md) コマンドを使用してsweepを初期化します。詳細については[Initialize sweeps](./initialize-sweeps.md)を参照してください。

```
wandb sweep config.yaml
```

整数値をcountフラグに渡して、試行するrunsの最大数を設定します。

```
NUM=10
SWEEPID="dtzl1o7u"
wandb agent --count $NUM $SWEEPID
```
  </TabItem>
</Tabs>