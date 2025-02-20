---
title: Start or stop a sweep agent
description: 1 台以上のマシンで W&B Sweep エージェントを開始または停止します。
menu:
  default:
    identifier: ja-guides-models-sweeps-start-sweep-agents
    parent: sweeps
weight: 5
---

Start a W&B Sweep を1台以上のマシンの1台以上のエージェントで開始します。W&B Sweep エージェントは、ハイパーパラメーターを取得するために、`wandb sweep` で W&B Sweep を初期化したときに起動した W&B サーバーに問い合わせ、これを使用してモデル トレーニングを実行します。

W&B Sweep エージェントを開始するには、W&B Sweep を初期化したときに返された W&B Sweep ID を指定します。W&B Sweep ID は次の形式です。

```bash
entity/project/sweep_ID
```

ここで：

* entity: W&B のユーザー名またはチーム名。
* project: W&B Run の出力を保存したいプロジェクトの名前。プロジェクトが指定されていない場合、run は「Uncategorized」プロジェクトに配置されます。
* sweep_ID: W&B によって生成された疑似乱数の一意の ID。

Jupyter Notebook や Python スクリプト内で W&B Sweep エージェントを開始する場合、W&B Sweep が実行する関数の名前を指定します。

次のコードスニペットは、W&B でエージェントを開始する方法を示しています。設定ファイルを既に持っており、W&B Sweep を既に初期化していることを前提としています。設定ファイルの定義方法について詳しくは、 [Define sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。 

{{< tabpane text=true >}}
{{% tab header="CLI" %}}
`sweep` を開始するには `wandb agent` コマンドを使用します。sweep を初期化したときに返された sweep ID を指定します。次のコードスニペットをコピーして貼り付け、`sweep_id` を自分の sweep ID に置き換えます。

```bash
wandb agent sweep_id
```
{{% /tab %}}
{{% tab header="Python script or notebook" %}}
W&B Python SDK ライブラリを使用して sweep を開始します。初期化された sweep ID を指定します。さらに、sweep が実行する関数の名前を指定します。

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```
{{% /tab %}}
{{< /tabpane >}}

### W&B エージェントの停止

{{% alert color="secondary" %}}
ランダムおよびベイズ探索は永遠に実行されます。コマンドライン内、Python スクリプト内、または [Sweeps UI]({{< relref path="./visualize-sweep-results.md" lang="ja" >}}) からプロセスを停止する必要があります。
{{% /alert %}}

オプションで、Sweep エージェントが試行する W&B Runs の数を指定します。以下のコードスニペットは、Jupyter Notebook、Python スクリプト内で CLI と共に最大数の [W&B Runs]({{< relref path="/ref/python/run.md" lang="ja" >}}) を設定する方法を示しています。

{{< tabpane text=true >}}
  {{% tab header="Python script or notebook" %}}
まず、sweep を初期化します。詳細については、 [Initialize sweeps]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

```
sweep_id = wandb.sweep(sweep_config)
```

次に、sweep ジョブを開始します。sweep 開始から生成された sweep ID を指定します。試行する最大 run 数を設定するために整数値を count パラメータに渡します。

```python
sweep_id, count = "dtzl1o7u", 10
wandb.agent(sweep_id, count=count)
```

{{% alert color="secondary" %}}
sweep agent が終了した後、同じスクリプトまたはノートブック内で新しい run を開始する場合は、新しい run を開始する前に `wandb.teardown()` を呼び出す必要があります。
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="CLI" %}}
まず、[`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) コマンドを使用して sweep を初期化します。詳細については、 [Initialize sweeps]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

```
wandb sweep config.yaml
```

試行する run の最大数を設定するために整数値を count フラグに渡します。

```python
NUM=10
SWEEPID="dtzl1o7u"
wandb agent --count $NUM $SWEEPID
```  
  {{% /tab %}}
{{< /tabpane >}}