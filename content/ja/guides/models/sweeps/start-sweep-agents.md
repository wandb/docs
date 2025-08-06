---
title: sweep エージェントの開始または停止
description: 1台以上のマシンで W&B sweep agent を開始または停止します。
menu:
  default:
    identifier: start-sweep-agents
    parent: sweeps
weight: 5
---

W&B Sweep を 1 台以上のマシン上で 1 つ以上のエージェントで開始します。W&B Sweep エージェントは Sweep を初期化したときに立ち上がった W&B サーバーへ問い合わせ、ハイパーパラメーターを取得して、モデルのトレーニングを実行します。

W&B Sweep エージェントを開始するには、Sweep の初期化時に返される W&B Sweep ID を指定します。W&B Sweep ID の形式は次の通りです。

```bash
entity/project/sweep_ID
```

各要素の意味は以下の通りです：

* entity: あなたの W&B ユーザー名またはチーム名
* project: この W&B Run の出力を保存したいプロジェクト名。指定しない場合、run は「Uncategorized」プロジェクトに保存されます。
* sweep_ID: W&B により生成される擬似ランダムな固有 ID

Jupyter Notebook や Python スクリプト内で W&B Sweep エージェントを開始する場合は、W&B Sweep が実行する関数名を指定してください。

以下のコードスニペットは、W&B でエージェントを起動する方法を示しています。ここでは、すでに設定ファイルを用意し、Sweep の初期化も完了していると仮定しています。設定ファイルの定義方法については [Define sweep configuration]({{< relref "/guides/models/sweeps/define-sweep-configuration/" >}}) を参照してください。

{{< tabpane text=true >}}
{{% tab header="CLI" %}}
`sweep` を開始するには、`wandb agent` コマンドを使います。Sweep 初期化時に返された Sweep ID を指定してください。以下のコードスニペットの `sweep_id` をご自身の Sweep ID に置き換えて使用します。

```bash
wandb agent sweep_id
```
{{% /tab %}}
{{% tab header="Python script or notebook" %}}
W&B Python SDK ライブラリを使って Sweep を開始します。Sweep 初期化時に返された Sweep ID、さらに Sweep が実行する関数名を指定してください。

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```
{{% /tab %}}
{{< /tabpane >}}



### W&B agent を停止する

{{% alert color="secondary" %}}
ランダムおよびベイズ探索はデフォルトで無限に実行されます。コマンドラインやご自身の Python スクリプト、または [Sweeps UI]({{< relref "./visualize-sweep-results.md" >}}) からプロセスを停止する必要があります。
{{% /alert %}}

オプションで、Sweep agent が実行する W&B Run の最大回数を指定することもできます。以下のコードスニペットは、CLI または Jupyter Notebook、Python スクリプト内で [W&B Run]({{< relref "/ref/python/sdk/classes/run.md" >}}) の最大試行数を設定する方法を示します。

{{< tabpane text=true >}}
  {{% tab header="Python script or notebook" %}}
まず、sweep を初期化します。詳細は [Initialize sweeps]({{< relref "./initialize-sweeps.md" >}}) を参照してください。

```
sweep_id = wandb.sweep(sweep_config)
```

次に、sweep ジョブを開始します。sweep の初期化時に生成された sweep ID を指定し、`count` パラメータに整数値を渡すことで、試行する run の最大数を設定できます。

```python
sweep_id, count = "dtzl1o7u", 10
wandb.agent(sweep_id, count=count)
```

{{% alert color="secondary" %}}
sweep agent の処理が完了した後、同じスクリプトやノートブック内で新しい run を開始する場合は、新しい run を開始する前に `wandb.teardown()` を呼び出してください。
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="CLI" %}}
まず、[`wandb sweep`]({{< relref "/ref/cli/wandb-sweep.md" >}}) コマンドで sweep を初期化します。詳細は [Initialize sweeps]({{< relref "./initialize-sweeps.md" >}}) をご覧ください。

```
wandb sweep config.yaml
```

`--count` フラグに整数値を渡すことで、試行する run の最大数を設定できます。

```python
NUM=10
SWEEPID="dtzl1o7u"
wandb agent --count $NUM $SWEEPID
```  
  {{% /tab %}}
{{< /tabpane >}}