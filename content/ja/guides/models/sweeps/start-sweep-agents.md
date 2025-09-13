---
title: sweep agent を開始または停止する
description: 1 台以上のマシン上で W&B Sweep Agent を起動または停止します。
menu:
  default:
    identifier: ja-guides-models-sweeps-start-sweep-agents
    parent: sweeps
weight: 5
---

1 台以上のマシン上の 1 つ以上の sweep agent で W&B Sweep を開始します。W&B Sweep agent は、W&B Sweep (`wandb sweep)` を初期化したときに起動した W&B サーバーにハイパーパラメーターを問い合わせ、それらを使って モデルトレーニング を実行します。

W&B Sweep agent を開始するには、W&B Sweep を初期化したときに返された W&B Sweep ID を指定します。W&B Sweep ID の形式は次のとおりです:

```bash
entity/project/sweep_ID
```

各フィールドの意味:

* entity: あなたの W&B のユーザー名またはチーム名。
* project: W&B Run の出力を保存したい Project の名前。project を指定しない場合、run は "Uncategorized" Project に配置されます。
* sweep_ID: W&B によって生成される疑似ランダムな一意の ID。

Jupyter ノートブックや Python スクリプト内で W&B Sweep agent を開始する場合は、W&B Sweep が実行する関数名を指定してください。

以下の コードスニペット は、W&B で sweep agent を開始する方法を示します。設定ファイルが既にあり、W&B Sweep を初期化済みであると仮定します。設定ファイルの定義方法については、[sweep configuration を定義する]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。

{{< tabpane text=true >}}
{{% tab header="CLI" %}}
sweep を開始するには `wandb agent` コマンドを使用します。sweep を初期化したときに返された sweep ID を指定してください。以下の コードスニペット をコピーして貼り付け、`sweep_id` をあなたの sweep ID に置き換えてください:

```bash
wandb agent sweep_id
```
{{% /tab %}}
{{% tab header="Python スクリプトまたはノートブック" %}}
W&B の Python SDK を使って sweep を開始できます。sweep を初期化したときに返された sweep ID を指定してください。加えて、その sweep が実行する関数名も指定します。

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```

{{% alert color="secondary" title="マルチプロセッシング" %}}
Python 標準ライブラリの `multiprocessing` や PyTorch の `pytorch.multiprocessing` パッケージを使用する場合は、`wandb.agent()` と `wandb.sweep()` の呼び出しを `if __name__ == '__main__':` で囲む必要があります。例えば:

```python
if __name__ == '__main__':
    wandb.agent(sweep_id="<sweep_id>", function="<function>", count="<count>")
```

この慣習で コード をラップすることで、スクリプトが直接実行されたときにのみ実行され、ワーカー プロセスでモジュールとしてインポートされたときには実行されないことを保証します。

マルチプロセッシングの詳細は、[Python 標準ライブラリ `multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods) または [PyTorch `multiprocessing`](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html#asynchronous-multiprocess-training-e-g-hogwild) を参照してください。`if __name__ == '__main__':` という慣習については https://realpython.com/if-name-main-python/ を参照してください。
{{% /alert %}}

{{% /tab %}}
{{< /tabpane >}}



### W&B エージェントを停止する

{{% alert color="secondary" %}}
ランダム探索と ベイズ探索 は無期限に実行されます。コマンドライン、Python スクリプト内、または [Sweeps UI]({{< relref path="./visualize-sweep-results.md" lang="ja" >}}) から プロセス を停止する必要があります。
{{% /alert %}}

任意で、sweep agent が試行する W&B Runs の数を指定できます。以下の コードスニペット は、CLI および Jupyter ノートブックや Python スクリプト内で、[W&B Runs]({{< relref path="/ref/python/sdk/classes/run.md" lang="ja" >}}) の最大数を設定する方法を示します。

{{< tabpane text=true >}}
  {{% tab header="Python スクリプトまたはノートブック" %}}
まず、sweep を初期化します。詳細は [Sweeps を初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

```
sweep_id = wandb.sweep(sweep_config)
```

次に、sweep ジョブを開始します。sweep の初期化で生成された sweep ID を指定します。試行する run の最大数を設定するには、count パラメータに整数の 値 を渡します。

```python
sweep_id, count = "dtzl1o7u", 10
wandb.agent(sweep_id, count=count)
```

{{% alert color="secondary" %}}
同じスクリプトやノートブック内で、sweep agent の終了後に新しい run を開始する場合は、その新しい run を始める前に `wandb.teardown()` を呼び出してください。
{{% /alert %}}
  {{% /tab %}}
  {{% tab header="CLI" %}}
まず、[`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) コマンドで sweep を初期化します。詳細は [Sweeps を初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

```
wandb sweep config.yaml
```

試行する run の最大数を設定するには、count フラグに整数の 値 を指定します。

```python
NUM=10
SWEEPID="dtzl1o7u"
wandb agent --count $NUM $SWEEPID
```
  {{% /tab %}}
{{< /tabpane >}}