---
title: sweep エージェントを開始または停止する
description: 1 台または複数のマシン上で W&B Sweep Agent を開始または停止します。
menu:
  default:
    identifier: ja-guides-models-sweeps-start-sweep-agents
    parent: sweeps
weight: 5
---

W&B Sweep を 1 台以上のマシン上の 1 台以上のエージェントで開始します。W&B Sweep エージェントは、ハイパーパラメーターを取得するために W&B Sweep を初期化したときにローンンチされた W&B サーバーにクエリを送り、それらを使用してモデル トレーニングを実行します。

W&B Sweep エージェントを開始するには、W&B Sweep を初期化したときに返された W&B Sweep ID を指定します。W&B Sweep ID の形式は次のとおりです。

```bash
entity/project/sweep_ID
```

ここで:

* entity: W&B のユーザー名またはチーム名。
* project: W&B Run の出力を保存したいプロジェクトの名前。プロジェクトが指定されていない場合、run は「未分類」プロジェクトに置かれます。
* sweep_ID: W&B によって生成される疑似ランダムな一意の ID。

Jupyter ノートブックまたは Python スクリプト内で W&B Sweep エージェントを開始する場合、W&B Sweep が実行する関数の名前を指定します。

次のコードスニペットは、W&B でエージェントを開始する方法を示しています。ここでは、既に設定ファイルを持っており、W&B Sweep を初期化済みであると仮定しています。設定ファイルを定義する方法について詳しくは、[Define sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。

{{< tabpane text=true >}}
{{% tab header="CLI" %}}
`sweep agent` コマンドを使用してスイープを開始します。スイープを初期化するときに返されたスイープ ID を指定します。以下のコードスニペットをコピーして貼り付け、`sweep_id` をスイープ ID に置き換えてください:

```bash
wandb agent sweep_id
```
{{% /tab %}}
{{% tab header="Python script or notebook" %}}
W&B Python SDK ライブラリを使用してスイープを開始します。スイープを初期化するときに返されたスイープ ID を指定します。さらに、スイープが実行する関数の名前も指定します。

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```
{{% /tab %}}
{{< /tabpane >}}

### W&B エージェントを停止

{{% alert color="secondary" %}}
ランダムおよびベイズ探索は永遠に実行されます。プロセスをコマンドライン、Python スクリプト内、または [Sweeps UI]({{< relref path="./visualize-sweep-results.md" lang="ja" >}}) から停止する必要があります。
{{% /alert %}}

オプションで、Sweep agent が試みるべき W&B Runs の数を指定します。以下のコードスニペットは、Jupyter ノートブック、Python スクリプト内で最大の [W&B Runs]({{< relref path="/ref/python/run.md" lang="ja" >}}) 数を設定する方法を示しています。

{{< tabpane text=true >}}
  {{% tab header="Python script or notebook" %}}
まず、スイープを初期化します。詳細は [Initialize sweeps]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

```
sweep_id = wandb.sweep(sweep_config)
```

次にスイープジョブを開始します。スイープ初期化から生成されたスイープ ID を提供します。試行する run の最大数を設定するために count パラメータに整数値を渡します。

```python
sweep_id, count = "dtzl1o7u", 10
wandb.agent(sweep_id, count=count)
```

{{% alert color="secondary" %}}
スイープエージェントが終了した後に新しい run を同じスクリプトまたはノートブック内で開始する場合は、新しい run を開始する前に `wandb.teardown()` を呼び出す必要があります。
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="CLI" %}}
まず、[`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) コマンドでスイープを初期化します。詳細は [Initialize sweeps]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

```
wandb sweep config.yaml
```

試行する run の最大数を設定するために、count フラグに整数値を渡します。

```python
NUM=10
SWEEPID="dtzl1o7u"
wandb agent --count $NUM $SWEEPID
```  
  {{% /tab %}}
{{< /tabpane >}}