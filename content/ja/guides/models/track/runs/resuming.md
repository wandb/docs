---
title: run を再開する
description: 一時停止または終了した W&B Run を再開する
menu:
  default:
    identifier: ja-guides-models-track-runs-resuming
    parent: what-are-runs
---

run が停止またはクラッシュした場合にどのように振る舞うかを指定します。run を再開したり、自動再開を有効にするには、その run に関連付けられた一意の run ID を `id` パラメータに指定する必要があります:

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```

{{% alert %}}
W&B は、run を保存したい W&B Project の名前を指定することを推奨します。
{{% /alert %}}

W&B の振る舞いを決めるために、`resume` パラメータに次のいずれかの引数を渡してください。どの場合でも、まず W&B は run ID が既に存在するかどうかを確認します。 

|引数 | 説明 | run ID が存在する場合 | run ID が存在しない場合 | ユースケース |
| --- | --- | -- | --| -- |
| `"must"` | W&B は run ID で指定された run を必ず再開する必要がある。 | W&B は同じ run ID の run を再開します。 | W&B はエラーを発生させます。 | 同じ run ID を必ず使って run を再開したい場合。 |
| `"allow"`| run ID が存在する場合に W&B に run を再開させます。 | W&B は同じ run ID の run を再開します。 | W&B は指定した run ID で新しい run を初期化します。 | 既存の run を上書きせずに run を再開したい場合。 |
| `"never"`| run ID で指定された run の再開を W&B に許可しません。 | W&B はエラーを発生させます。 | W&B は指定した run ID で新しい run を初期化します。 | |




`resume="auto"` を指定して、W&B に run の再起動を自動的に試行させることもできます。ただし、同じディレクトリーで run を再起動する必要があります。詳しくは、[run の自動再開を有効にする]({{< relref path="#enable-runs-to-automatically-resume" lang="ja" >}}) を参照してください。

以下の例では、`<>` で囲まれた値を自分の値に置き換えてください。

## 同じ run ID を必ず使って run を再開する
run が停止、クラッシュ、または失敗した場合、同じ run ID を使って再開できます。そのためには、run を初期化して次を指定します:

* `resume` パラメータを `"must"`（`resume="must"`）に設定する 
* 停止またはクラッシュした run の run ID を指定する




次のコードスニペットは W&B Python SDK での方法を示します:

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="must")
```

{{% alert color="secondary" %}}
複数のプロセスが同じ `id` を同時に使用すると、予期しない結果が発生します。 


複数のプロセスの管理方法については、[分散トレーニングをログする]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}}) を参照してください。 
{{% /alert %}}

## 既存の run を上書きせずに run を再開する
停止またはクラッシュした run を、既存の run を上書きせずに再開します。プロセスが正常に終了しなかった場合に特に役立ちます。次に W&B を開始したとき、W&B は最後のステップからログを開始します。

W&B で run を初期化する際に `resume` パラメータを `"allow"`（`resume="allow"`）に設定します。停止またはクラッシュした run の run ID を指定してください。次のコードスニペットは W&B Python SDK での方法を示します:

```python
import wandb

run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="allow")
```


## run の自動再開を有効にする 
次のコードスニペットは、Python SDK または環境変数で run の自動再開を有効にする方法を示します。 

{{< tabpane text=true >}}
  {{% tab header="W&B Python SDK" %}}
次のコードスニペットは、Python SDK で W&B の run ID を指定する方法を示します。 

`<>` で囲まれた値を自分の値に置き換えてください:

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```  
  {{% /tab %}}
  {{% tab header="シェルスクリプト" %}}

次の例は、bash スクリプトで W&B の `WANDB_RUN_ID` 変数を指定する方法を示します: 

```bash title="run_experiment.sh"
RUN_ID="$1"

WANDB_RESUME=allow WANDB_RUN_ID="$RUN_ID" python eval.py
```
ターミナル内で、W&B の run ID とともにこのシェルスクリプトを実行できます。次のコードスニペットでは run ID `akj172` を渡しています: 

```bash
sh run_experiment.sh akj172 
```

{{% /tab %}}
{{< /tabpane >}}


{{% alert color="secondary" %}}
自動再開は、失敗したプロセスと同じファイルシステム上でプロセスが再起動される場合にのみ機能します。 
{{% /alert %}}


例えば、`Users/AwesomeEmployee/Desktop/ImageClassify/training/` というディレクトリーで `train.py` という Python スクリプトを実行するとします。`train.py` の中で、自動再開を有効にする run を作成します。その後、training スクリプトが停止したとします。この run を再開するには、`Users/AwesomeEmployee/Desktop/ImageClassify/training/` 内で `train.py` スクリプトを再実行する必要があります。


{{% alert %}}
ファイルシステムを共有できない場合は、`WANDB_RUN_ID` 環境変数を指定するか、W&B Python SDK で run ID を渡してください。run ID の詳細は「What are runs?」ページの [カスタム run ID]({{< relref path="./#custom-run-ids" lang="ja" >}}) セクションを参照してください。
{{% /alert %}}





## プリエンプト可能な Sweeps の run を再開する
中断された [sweep]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) run を自動で再キューします。これは、プリエンプトが発生する可能性がある計算環境（プリエンプト可能キューの SLURM ジョブ、EC2 スポットインスタンス、Google Cloud の preemptible VM など）で sweep agent を実行する場合に特に有用です。

中断された sweep run を自動で再キューするには、[`mark_preempting`]({{< relref path="/ref/python/sdk/classes/run#mark_preempting" lang="ja" >}}) 関数を使用します。例:

```python
run = wandb.init()  # run を初期化
run.mark_preempting()
```
以下の表は、sweep run の終了ステータスに基づき W&B が run をどのように処理するかを示します。

|ステータス| 振る舞い |
|------| ---------|
|ステータスコード 0| run は正常終了と見なされ、再キューされません。  |
|非 0 のステータス| W&B はその run を該当 sweep の run キューに自動で追加します。|
|ステータスなし| run は sweep の run キューに追加されます。sweep agent は、キューが空になるまで run を消費します。キューが空になると、sweep キューは sweep の探索アルゴリズムに基づいて新しい run の生成を再開します。|