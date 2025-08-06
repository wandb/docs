---
title: run を再開する
description: 一時停止または終了した W&B Run を再開する
menu:
  default:
    identifier: ja-guides-models-track-runs-resuming
    parent: what-are-runs
---

run が停止またはクラッシュした場合にどのように振る舞うかを指定します。run を再開したり自動再開を有効にするには、その run に関連付けられた一意の run ID を `id` パラメータで指定する必要があります。

```python
run = wandb.init(entity="<entity>", \
        project="<project>", id="<run ID>", resume="<resume>")
```

{{% alert %}}
W&B で run を保存したい Project の名前を指定することをおすすめします。
{{% /alert %}}

`resume` パラメータには、W&B の振る舞いを決める以下のいずれかの引数を渡してください。いずれの場合も、最初に W&B が run ID の存在をチェックします。

|引数 | 説明 | Run ID が存在する場合| Run ID が存在しない場合 | ユースケース |
| --- | --- | -- | --| -- |
| `"must"` | 指定した run ID の run を必ず再開します。 | 同じ run ID の run を再開します。 | エラーが発生します。 | 必ず同じ run ID で run を再開したい場合。  |
| `"allow"`| run ID が存在すれば W&B が run を再開します。 | 同じ run ID の run を再開します。 | 指定した run ID で新しい run を初期化します。 | 既存の run を上書きせずに再開したい場合。 |
| `"never"`| 指定した run ID の run は絶対に再開しません。 | エラーが発生します。 | 指定した run ID で新しい run を初期化します。 | |


また、`resume="auto"` を指定することで、W&B に run の自動再開を任せることもできます。ただし、run を同じディレクトリから再起動する必要があります。詳しくは [run の自動再開を有効化する]({{< relref path="#enable-runs-to-automatically-resume" lang="ja" >}}) セクションをご覧ください。

以下の例で登場する `<>` で囲まれた値は、すべてご自身の値に置き換えてください。

## 同じ run ID で必ず run を再開する

run が停止やクラッシュ、失敗した場合に、同じ run ID で再開できます。これを行うには、run を初期化する際に下記を指定します。

* `resume` パラメータに `"must"` (`resume="must"`) を設定
* 停止またはクラッシュした run の run ID を指定

以下は W&B Python SDK でのコード例です。

```python
run = wandb.init(entity="<entity>", \
        project="<project>", id="<run ID>", resume="must")
```

{{% alert color="secondary" %}}
同じ `id` を複数プロセスで同時に使用すると、予期しない結果につながります。

複数プロセスをどう取り扱うかについては、[分散トレーニング実験のログを記録する方法]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}}) をご参照ください。
{{% /alert %}}

## 既存の run を上書きせずに再開する

停止やクラッシュした run を既存の run を上書きせずに再開します。プロセスが正常終了しなかった場合など特に有用です。次回 W&B を起動すると、前回のステップからログを継続できます。

W&B で run を初期化する際、`resume` パラメータに `"allow"` (`resume="allow"`) を設定し、停止した run の run ID を指定してください。以下は Python SDK を使った例です。

```python
import wandb

run = wandb.init(entity="<entity>", \
        project="<project>", id="<run ID>", resume="allow")
```

## run の自動再開を有効化する

以下のコードスニペットは、Python SDK や環境変数を用いて run の自動再開を有効化する方法を示しています。

{{< tabpane text=true >}}
  {{% tab header="W&B Python SDK" %}}
以下は W&B run ID を Python SDK で指定する方法です。

`<>` で囲まれた値はご自身のものに置き換えてください。

```python
run = wandb.init(entity="<entity>", \
        project="<project>", id="<run ID>", resume="<resume>")
```  
  {{% /tab %}}
  {{% tab header="Shell script" %}}

次の例は、bash スクリプトで W&B の `WANDB_RUN_ID` 変数を指定する方法です。

```bash title="run_experiment.sh"
RUN_ID="$1"

WANDB_RESUME=allow WANDB_RUN_ID="$RUN_ID" python eval.py
```
ターミナル内で、W&B の run ID とともにシェルスクリプトを実行できます。以下のコード例では run ID `akj172` を渡しています。

```bash
sh run_experiment.sh akj172 
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert color="secondary" %}}
プロセスが失敗したプロセスと同じファイルシステム上で再起動される場合のみ、自動再開が機能します。
{{% /alert %}}

たとえば、`Users/AwesomeEmployee/Desktop/ImageClassify/training/` というディレクトリで `train.py` という Python スクリプトを実行し、その中で自動再開を有効にする run を作成したとします。その後、トレーニングスクリプトが停止した場合、この run を再開するには、再び同じディレクトリ `Users/AwesomeEmployee/Desktop/ImageClassify/training/` で `train.py` スクリプトを実行する必要があります。

{{% alert %}}
もしファイルシステムを共有できない場合は、`WANDB_RUN_ID` 環境変数を指定するか、W&B Python SDK で run ID を指定してください。run ID の詳細は ["What are runs?" ページの Custom run IDs]({{< relref path="./#custom-run-ids" lang="ja" >}}) のセクションをご覧ください。
{{% /alert %}}

## プリエンプティブルな Sweeps の run を再開する

中断された [sweep]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) run を自動的に再キューできます。たとえば、SLURM のプリエンプティブルキュー上のジョブや、EC2 スポットインスタンス、Google Cloud のプリエンプティブル VM など、プリエンプションされうる計算環境で sweep agent を実行している場合に便利です。

[`mark_preempting`]({{< relref path="/ref/python/sdk/classes/run#mark_preempting" lang="ja" >}}) 関数を使うことで、中断された sweep run を自動で再キューできます。例：

```python
run = wandb.init()  # run を初期化
run.mark_preempting()
```
以下の表は、sweep run の終了ステータスに基づき W&B がどのように run を扱うかを示しています。

|ステータス| 振る舞い |
|------| ---------|
|ステータスコード 0| run は正常に終了したとみなされ、再キューされません。 |
|非ゼロのステータス| W&B が run を sweep に紐づく run キューに自動追加します。|
|ステータスなし| run が sweep run キューに追加されます。sweep agent はキューが空になるまで run を消化し、空になれば sweep キューから新しい run が sweep の探索アルゴリズムに基づいて生成され続けます。|