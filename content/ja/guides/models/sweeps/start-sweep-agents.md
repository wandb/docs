---
title: sweep agent を開始または停止する
description: 1 台以上のマシンで W&B Sweep Agent を開始または停止します。
menu:
  default:
    identifier: ja-guides-models-sweeps-start-sweep-agents
    parent: sweeps
weight: 5
---

W&B Sweep を 1 台以上のエージェント、1 台以上のマシン上で開始します。W&B Sweep エージェントは、Sweep を初期化した際に起動した W&B サーバーにハイパーパラメーターの問い合わせを行い、それらを使ってモデルのトレーニングを実行します。

W&B Sweep エージェントを開始するには、Sweep を初期化した際に返される W&B Sweep ID を指定します。W&B Sweep ID の形式は以下のとおりです。

```bash
entity/project/sweep_ID
```

それぞれの意味は次の通りです。

* entity: あなたの W&B ユーザー名またはチーム名
* project: W&B Run の出力を保存したいプロジェクト名。指定しない場合は "Uncategorized" プロジェクトに保存されます。
* sweep_ID: W&B が生成する疑似ランダムなユニーク ID

Jupyter Notebook や Python スクリプトの中で W&B Sweep エージェントを開始する場合は、Sweep が実行する関数名を指定してください。

以下のコードスニペットでは、W&B でエージェントを開始する方法を示します。ここでは、すでに設定ファイルの準備と Sweep の初期化が完了しているものとします。設定ファイルの定義方法については [Sweep 設定の定義]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) をご覧ください。

{{< tabpane text=true >}}
{{% tab header="CLI" %}}
`sweep` を開始するには `wandb agent` コマンドを実行します。Sweep 初期化時に返される Sweep ID を指定してください。下記のコードスニペットをコピーして、`sweep_id` をご自身の Sweep ID に置き換えて実行します。

```bash
wandb agent sweep_id
```
{{% /tab %}}
{{% tab header="Python script or notebook" %}}
W&B の Python SDK ライブラリを使って sweep を開始します。Sweep 初期化時に返される Sweep ID を指定します。さらに、エージェントで実行したい関数名も指定してください。

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```
{{% /tab %}}
{{< /tabpane >}}



### W&B エージェントの停止

{{% alert color="secondary" %}}
ランダム探索およびベイズ探索は終了しないため、コマンドラインや Python スクリプト、または [Sweeps UI]({{< relref path="./visualize-sweep-results.md" lang="ja" >}}) からプロセスを手動で停止する必要があります。
{{% /alert %}}

必要に応じて、Sweep エージェントが試行する W&B Run の数を指定することも可能です。以下のコードスニペットは、CLI や Jupyter Notebook、Python スクリプト内で [W&B Run]({{< relref path="/ref/python/sdk/classes/run.md" lang="ja" >}}) の最大数を設定する方法を示しています。

{{< tabpane text=true >}}
  {{% tab header="Python script or notebook" %}}
まず、Sweep を初期化します。詳細は [Sweeps の初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

```
sweep_id = wandb.sweep(sweep_config)
```

次に Sweep ジョブを開始します。Sweep の開始時に生成された Sweep ID を指定してください。`count` パラメータに整数値を渡すことで、試行する Run の最大数を設定できます。

```python
sweep_id, count = "dtzl1o7u", 10
wandb.agent(sweep_id, count=count)
```

{{% alert color="secondary" %}}
Sweep エージェントの完了後、同じスクリプトやノートブック内で新たに Run を開始する場合は、`wandb.teardown()` を呼び出してから新しい Run を開始してください。
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="CLI" %}}
まずは [`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) コマンドで Sweep を初期化します。詳細は [Sweeps の初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

```
wandb sweep config.yaml
```

最大で何回 Run を試行するかは、`count` フラグに整数値を渡して設定できます。

```python
NUM=10
SWEEPID="dtzl1o7u"
wandb agent --count $NUM $SWEEPID
```  
  {{% /tab %}}
{{< /tabpane >}}