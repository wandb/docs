---
title: run を再開する
description: 一時停止または終了した W&B Run を再開する
menu:
  default:
    identifier: resuming
    parent: what-are-runs
---

run が停止またはクラッシュした場合にどのような挙動を取るかを指定できます。run を再開または自動再開を有効にするには、その run に関連付けられている一意の run ID を `id` パラメータで指定してください。

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```

{{% alert %}}
W&B では、run を保存したい W&B Project の名前を指定することを推奨しています。
{{% /alert %}}

W&B の振る舞いを決めるために、`resume` パラメータに次のいずれかの引数を渡します。いずれの場合も、まず W&B はその run ID が既に存在するかを確認します。

|引数 | 説明 | run ID が存在する場合 | run ID が存在しない場合 | ユースケース |
| --- | --- | -- | -- | -- |
| `"must"` | 指定した run ID の run を必ず再開します。 | 同じ run ID で run を再開します。 | エラーが発生します。 | 必ず同じ run ID で run を再開したい場合。 |
| `"allow"`| run ID が存在する場合に限り run を再開します。 | 同じ run ID で run を再開します。 | 指定された run ID で新しい run を初期化します。 | 既存の run を上書きせずに再開したい場合。 |
| `"never"`| 指定された run ID の run を再開しません。 | エラーが発生します。 | 指定された run ID で新しい run を初期化します。 | |

また、`resume="auto"` を指定することで、W&B が自動的に run の再開を試みることもできます。ただし、同じディレクトリーから run を再開する必要があります。詳細は [run の自動再開を有効にする]({{< relref "#enable-runs-to-automatically-resume" >}}) セクションをご覧ください。

以下の例では、`<>` で囲まれた値をあなた自身の値に置き換えてください。

## 同じ run ID を使って run を再開する
run が停止、クラッシュ、または失敗した場合、同じ run ID を使って再開できます。そのためには、run を初期化して以下を指定してください。

* `resume` パラメータを `"must"` (`resume="must"`) に設定する
* 停止またはクラッシュした run の run ID を指定する

以下のコードスニペットは、W&B Python SDK でこの設定を行う例です。

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="must")
```

{{% alert color="secondary" %}}
複数のプロセスが同じ `id` を同時に使用すると、予期しない結果になる場合があります。

複数プロセスの管理方法については [分散トレーニング実験のログ取得]({{< relref "/guides/models/track/log/distributed-training.md" >}}) をご覧ください。
{{% /alert %}}

## 既存の run を上書きせずに run を再開する
停止やクラッシュした run を、既存 run を上書きせずに再開します。プロセスが正常に終了しなかった場合に特に有用です。次回 W&B を起動すると、W&B は最後のステップからログを再開します。

run を初期化する際に `resume` パラメータを `"allow"` (`resume="allow"`) に設定してください。また、停止またはクラッシュした run の run ID を指定します。以下はその設定例です。

```python
import wandb

run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="allow")
```

## run の自動再開を有効にする
以下のコードスニペットは、Python SDK や環境変数を使って run の自動再開を有効にする方法の例です。

{{< tabpane text=true >}}
  {{% tab header="W&B Python SDK" %}}
以下のコードスニペットは、Python SDK で W&B run ID を指定する方法を示しています。

`<>` で囲まれた部分は適切な値に置き換えてください。

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```  
  {{% /tab %}}
  {{% tab header="Shell script" %}}

以下の例は、bash スクリプトで W&B の `WANDB_RUN_ID` 変数を指定する方法です。

```bash title="run_experiment.sh"
RUN_ID="$1"

WANDB_RESUME=allow WANDB_RUN_ID="$RUN_ID" python eval.py
```
ターミナルからは、以下のように W&B run ID を指定してシェルスクリプトを実行できます。次のスニペットは run ID `akj172` を渡す例です。

```bash
sh run_experiment.sh akj172 
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert color="secondary" %}}
自動再開は、失敗したプロセスと同じファイルシステム上でプロセスを再実行した場合のみ機能します。
{{% /alert %}}

例えば、`Users/AwesomeEmployee/Desktop/ImageClassify/training/` というディレクトリーで `train.py` という Python スクリプトを実行しているケースを考えます。`train.py` では自動再開を有効にした run が作成されます。その後、トレーニングスクリプトが停止したとします。この run を再開するには、`Users/AwesomeEmployee/Desktop/ImageClassify/training/` 内で `train.py` を再度実行する必要があります。

{{% alert %}}
もしファイルシステムを共有できない場合は、`WANDB_RUN_ID` 環境変数を指定するか、W&B Python SDK の run ID オプションで直接 run ID を渡してください。run ID について詳しくは、「What are runs?」ページの [カスタム run ID]({{< relref "./#custom-run-ids" >}}) セクションをご覧ください。
{{% /alert %}}

## プリエンプティブルな Sweeps run を再開する
中断された [sweep]({{< relref "/guides/models/sweeps/" >}}) run を自動的に再キューイングします。これは、SLURM のプリエンプティブルキューや EC2 のスポットインスタンス、Google Cloud のプリエンプティブル VM など、プリエンプションが発生する可能性のある計算環境で sweep agent を実行している場合に特に便利です。

中断された sweep run を自動的に再キューイングするには、[`mark_preempting`]({{< relref "/ref/python/sdk/classes/run#mark_preempting" >}}) 関数を使用します。例:

```python
run = wandb.init()  # run を初期化
run.mark_preempting()
```
以下の表は、sweep run の終了ステータスに応じて W&B がどのように run を扱うかをまとめたものです。

|ステータス| 振る舞い |
|------| ---------|
|ステータスコード 0| run は正常に終了したと見なされ、再キューされません。 |
|非ゼロステータス| W&B は run を sweep に関連付けられた run キューに自動的に追加します。|
|ステータスなし| run は sweep run キューに追加されます。sweep agent はキューが空になるまで run を消化します。キューが空になると、sweep の探索アルゴリズムに基づいて新たな run が生成されます。|