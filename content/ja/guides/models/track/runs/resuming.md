---
title: Resume a run
description: 一時停止または終了した W&B Run を再開する
menu:
  default:
    identifier: ja-guides-models-track-runs-resuming
    parent: what-are-runs
---

run が停止またはクラッシュした場合に、run がどのように振る舞うかを指定します。run を再開するか、run が自動的に再開できるようにするには、`id` パラメータに対して、その run に関連付けられた一意の run ID を指定する必要があります。

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```

{{% alert %}}
W&B は、run を保存する W&B の Project 名を提供することを推奨します。
{{% /alert %}}

W&B がどのように応答するかを決定するために、次の引数のいずれかを `resume` パラメータに渡します。いずれの場合も、W&B は最初に run ID がすでに存在するかどうかを確認します。

|引数 | 説明 | Run ID が存在する場合 | Run ID が存在しない場合 | ユースケース |
| --- | --- | -- | --| -- |
| `"must"` | W&B は、run ID で指定された run を再開する必要があります。 | W&B は同じ run ID で run を再開します。 | W&B はエラーを発生させます。 | 同じ run ID を使用する必要がある run を再開します。 |
| `"allow"`| W&B は、run ID が存在する場合に run を再開することを許可します。 | W&B は同じ run ID で run を再開します。 | W&B は、指定された run ID で新しい run を初期化します。 | 既存の run を上書きせずに run を再開します。 |
| `"never"`| W&B は、run ID で指定された run を再開することを許可しません。 | W&B はエラーを発生させます。 | W&B は、指定された run ID で新しい run を初期化します。 | |

`resume="auto"` を指定して、W&B が自動的に run の再起動を試みるようにすることもできます。ただし、同じディレクトリーから run を再起動する必要があります。[run を自動的に再開できるようにする]({{< relref path="#enable-runs-to-automatically-resume" lang="ja" >}}) セクションを参照してください。

以下のすべての例で、`<>` で囲まれた値を独自の値に置き換えてください。

## 同じ run ID を使用する必要がある run を再開する
run が停止、クラッシュ、または失敗した場合、同じ run ID を使用して再開できます。これを行うには、run を初期化し、以下を指定します。

* `resume` パラメータを `"must"` (`resume="must"`) に設定します。
* 停止またはクラッシュした run の run ID を指定します。

次のコードスニペットは、W&B Python SDK でこれを実現する方法を示しています。

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="must")
```

{{% alert color="secondary" %}}
複数のプロセスが同じ `id` を同時に使用すると、予期しない結果が発生します。

複数のプロセスを管理する方法の詳細については、[分散トレーニング Experiments のログ記録]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

## 既存の run を上書きせずに run を再開する
既存の run を上書きせずに、停止またはクラッシュした run を再開します。これは、プロセスが正常に終了しない場合に特に役立ちます。次回 W&B を起動すると、W&B は最後のステップからログの記録を開始します。

W&B で run を初期化するときに、`resume` パラメータを `"allow"` (`resume="allow"`) に設定します。停止またはクラッシュした run の run ID を指定します。次のコードスニペットは、W&B Python SDK でこれを実現する方法を示しています。

```python
import wandb

run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="allow")
```

## run を自動的に再開できるようにする
次のコードスニペットは、Python SDK または環境変数を使用して、run を自動的に再開できるようにする方法を示しています。

{{< tabpane text=true >}}
  {{% tab header="W&B Python SDK" %}}
次のコードスニペットは、Python SDK で W&B の run ID を指定する方法を示しています。

`<>` で囲まれた値を独自の値に置き換えてください。

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```  
  {{% /tab %}}
  {{% tab header="Shell script" %}}

次の例は、bash スクリプトで W&B の `WANDB_RUN_ID` 変数を指定する方法を示しています。

```bash title="run_experiment.sh"
RUN_ID="$1"

WANDB_RESUME=allow WANDB_RUN_ID="$RUN_ID" python eval.py
```
ターミナル内で、W&B の run ID と共にシェルスクリプトを実行できます。次のコードスニペットは、run ID `akj172` を渡します。

```bash
sh run_experiment.sh akj172 
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert color="secondary" %}}
自動再開は、プロセスが失敗したプロセスと同じファイルシステム上で再起動された場合にのみ機能します。
{{% /alert %}}

たとえば、`Users/AwesomeEmployee/Desktop/ImageClassify/training/` というディレクトリーで `train.py` という Python スクリプトを実行するとします。`train.py` 内で、スクリプトは自動再開を有効にする run を作成します。次に、トレーニングスクリプトが停止したとします。この run を再開するには、`Users/AwesomeEmployee/Desktop/ImageClassify/training/` 内で `train.py` スクリプトを再起動する必要があります。

{{% alert %}}
ファイルシステムを共有できない場合は、`WANDB_RUN_ID` 環境変数を指定するか、W&B Python SDK で run ID を渡します。run ID の詳細については、「run とは何ですか?」ページの [カスタム run ID]({{< relref path="./#custom-run-ids" lang="ja" >}}) セクションを参照してください。
{{% /alert %}}

## プリエンプティブ Sweeps run の再開
中断された [sweep]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) run を自動的に再キューします。これは、プリエンプティブキューの SLURM ジョブ、EC2 スポットインスタンス、または Google Cloud プリエンプティブ VM など、プリエンプションの影響を受けるコンピューティング環境で sweep agent を実行する場合に特に役立ちます。

[`mark_preempting`]({{< relref path="/ref/python/run.md#mark_preempting" lang="ja" >}}) 関数を使用して、W&B が中断された sweep run を自動的に再キューできるようにします。たとえば、次のコードスニペット

```python
run = wandb.init()  # run を初期化します
run.mark_preempting()
```
次の表は、sweep run の終了ステータスに基づいて W&B が run をどのように処理するかを示しています。

|ステータス| 振る舞い |
|------| ---------|
|ステータスコード 0| Run は正常に終了したと見なされ、再キューされません。 |
|ゼロ以外のステータス| W&B は、run を sweep に関連付けられた run キューに自動的に追加します。|
|ステータスなし| Run は sweep run キューに追加されます。Sweep agent は、キューが空になるまで run キューから run を消費します。キューが空になると、sweep キューは sweep 検索アルゴリズムに基づいて新しい run の生成を再開します。|
