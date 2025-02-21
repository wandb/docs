---
title: Start or stop a sweep agent
description: 複数のマシン上で W&B Sweep Agent を起動または停止します。
menu:
  default:
    identifier: ja-guides-models-sweeps-start-sweep-agents
    parent: sweeps
weight: 5
---

複数のマシン上の1つまたは複数のエージェントで W&B Sweep を開始します。W&B Sweep エージェントは、ハイパーパラメータのために W&B Sweep ( `wandb sweep` ) を初期化したときに起動した W&B サーバーにクエリを実行し、それらを使用して モデルトレーニング を実行します。

W&B Sweep エージェントを開始するには、W&B Sweep を初期化したときに返された W&B Sweep ID を指定します。W&B Sweep ID の形式は次のとおりです。

```bash
entity/project/sweep_ID
```

以下にそれぞれの意味を示します。

* entity: あなたの W&B ユーザー名または Teams 名。
* project: W&B Run の出力を保存する project の名前。project が指定されていない場合、run は「未分類」の project に配置されます。
* sweep_ID: W&B によって生成された疑似ランダムな一意の ID。

Jupyter Notebook または Python スクリプト内で W&B Sweep エージェントを開始する場合は、W&B Sweep が実行する関数の名前を指定します。

以下の コードスニペット は、W&B でエージェントを開始する方法を示しています。設定ファイルが既にあって、W&B Sweep を初期化済みであると想定しています。設定ファイルを定義する方法の詳細については、[sweep configuration の定義]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。

{{< tabpane text=true >}}
{{% tab header="CLI" %}}
sweep を開始するには、`wandb agent` コマンドを使用します。sweep を初期化したときに返された sweep ID を指定します。以下の コードスニペット をコピーして貼り付け、`sweep_id` を自分の sweep ID に置き換えます。

```bash
wandb agent sweep_id
```
{{% /tab %}}
{{% tab header="Python script or notebook" %}}
W&B Python SDK ライブラリを使用して sweep を開始します。sweep を初期化したときに返された sweep ID を指定します。また、sweep が実行する関数の名前も指定します。

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```
{{% /tab %}}
{{< /tabpane >}}

### W&B エージェントの停止

{{% alert color="secondary" %}}
ランダム探索と ベイズ探索 は永久に実行されます。コマンドライン、Python スクリプト内、または [Sweeps UI]({{< relref path="./visualize-sweep-results.md" lang="ja" >}}) から プロセス を停止する必要があります。
{{% /alert %}}

オプションで、Sweep エージェントが試行する W&B Runs の数を指定します。次の コードスニペット は、CLI および Jupyter Notebook、Python スクリプト内で [W&B Runs]({{< relref path="/ref/python/run.md" lang="ja" >}}) の最大数を設定する方法を示しています。

{{< tabpane text=true >}}
  {{% tab header="Python script or notebook" %}}
まず、sweep を初期化します。詳細については、[sweep の初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

```
sweep_id = wandb.sweep(sweep_config)
```

次に、sweep ジョブを開始します。sweep の初期化から生成された sweep ID を指定します。試行するrun の最大数を設定するには、count パラメータに整数値を渡します。

```python
sweep_id, count = "dtzl1o7u", 10
wandb.agent(sweep_id, count=count)
```

{{% alert color="secondary" %}}
sweep agent の完了後、同じ スクリプト または ノートブック 内で新しい run を開始する場合は、新しい run を開始する前に `wandb.teardown()` を呼び出す必要があります。
{{% /alert %}}
  {{% /tab %}}
  {{% tab header="CLI" %}}
まず、[`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) コマンドで sweep を初期化します。詳細については、[sweep の初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

```
wandb sweep config.yaml
```

試行する run の最大数を設定するには、count フラグに整数値を渡します。

```python
NUM=10
SWEEPID="dtzl1o7u"
wandb agent --count $NUM $SWEEPID
```
  {{% /tab %}}
{{< /tabpane >}}
