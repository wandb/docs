---
description: 一時停止または終了した W&B Run の再開
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Resume Runs

<head>
  <title>Resume W&B Runs</title>
</head>

run が停止またはクラッシュした場合の挙動を指定します。run を再開または自動再開を有効にするには、その run に関連付けられた一意の run の ID を `id` パラメータに指定する必要があります:
```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```

:::tip
W&B は、run を保存する W&B Project の名前を指定することを推奨しています。
:::

次のいずれかの引数を `resume` パラメータに渡して、W&B がどのように対応すべきかを決定します。いずれの場合も、W&B はまず run ID が既に存在するかどうかを確認します。

|Argument | Description | Run ID exists| Run ID does not exist | Use case |
| --- | --- | -- | --| -- |
| `"must"` | W&B が run ID で指定された run を再開する必要があります。 | W&B は同じ run ID で run を再開します。 | W&B はエラーを発生させます。 | 同じ run ID を使用する必要がある run を再開します。 |
| `"allow"`| run ID が存在する場合、W&B は run の再開を許可します。 | W&B は同じ run ID で run を再開します。 | W&B は指定された run ID で新しい run を初期化します。 | 既存の run を上書きせずに run を再開します。 |
| `"never"`| run ID で指定された run を W&B が再開することを禁止します。 | W&B はエラーを発生させます。 | W&B は指定された run ID で新しい run を初期化します。 | |

また、`resume="auto"` を指定することで、W&B がユーザーの代わりに自動的に run を再起動するように試みることもできます。ただし、その run を同じディレクトリーから再起動する必要があります。詳細については、[Enable runs to automatically resume](#enable-runs-to-automatically-resume) セクションを参照してください。

以下の例では、`<>` で囲まれた値を自分の値に置き換えてください。

## 同じ run ID を使わなければいけない run の再開
停止、クラッシュ、または失敗した場合に、同じ run ID を使用して run を再開します。これを行うには、run を初期化し、以下を指定します:

* `resume` パラメータを `"must"` (`resume="must"`) に設定
* 停止またはクラッシュした run の run ID を提供

以下のコードスニペットは、W&B Python SDK を使用してこれを達成する方法を示しています:

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="must")
```

:::caution
複数のプロセスが同じ `id` を同時に使用すると予期しない結果が生じます。

複数のプロセスの管理方法の詳細については、[Log distributed training experiments](../track/log/distributed-training.md) を参照してください。
:::

## 既存の run を上書きせずに run を再開する
既存の run を上書きせずに停止またはクラッシュした run を再開します。これは特に、プロセスが正常に終了しない場合に役立ちます。次に W&B を開始したとき、W&B は最後のステップからログを開始します。

W&B を使って run を初期化するときに、`resume` パラメータを `"allow"` (`resume="allow"`) に設定します。停止またはクラッシュした run の run ID を提供します。以下のコードスニペットは、W&B Python SDK を使用してこれを達成する方法を示しています:

```python
import wandb

run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="allow")
```

## run を自動で再開するように有効化する
以下のコードスニペットは、Python SDK または環境変数を使用して run を自動で再開する方法を示しています。

<Tabs
  defaultValue="python"
  values={[
    {label: 'W&B Python SDK', value: 'python'},
    {label: 'Shell script', value: 'bash'},
  ]}>
  <TabItem value="python">

以下のコードスニペットは、Python SDK を使用して W&B run ID を指定する方法を示しています。

`<>` で囲まれた値を自分の値に置き換えてください:

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```

  </TabItem>
  <TabItem value="bash">

以下の例は、bash スクリプト内で W&B `WANDB_RUN_ID` 変数を指定する方法を示しています:

```bash title="run_experiment.sh"
RUN_ID="$1"

WANDB_RESUME=allow WANDB_RUN_ID="$RUN_ID" python eval.py
```
ターミナル内で、W&B run ID と共にシェルスクリプトを実行できます。以下のコードスニペットは run ID `akj172` を渡しています:

```bash
sh run_experiment.sh akj172 
```

  </TabItem>
</Tabs>

:::important
プロセスが失敗したプロセスと同じファイルシステム上で再起動された場合にのみ、自動再開が機能します。
:::

例えば、`Users/AwesomeEmployee/Desktop/ImageClassify/training/` というディレクトリーで `train.py` という Python スクリプトを実行したとしましょう。`train.py` 内では、自動再開を有効にする run を作成します。次に、トレーニングスクリプトが停止したと仮定します。この run を再開するには、`Users/AwesomeEmployee/Desktop/ImageClassify/training/` 内で `train.py` スクリプトを再起動する必要があります。

:::tip
ファイルシステムを共有できない場合は、W&B Python SDK を使用して `WANDB_RUN_ID` 環境変数を指定するか、run ID を渡してください。run ID の詳細については、「What are runs?」ページの [Create a run](./intro.md#create-a-run) セクションを参照してください。
:::

## 中断された Sweep の自動再キュー
中断された [sweep](../sweeps/intro.md) runs を自動的に再キューします。これは、SLURM ジョブの中断可能キュー、EC2 スポットインスタンス、または Google Cloud の中断可能 VM など、中断される可能性がある計算環境で sweep エージェントを実行する場合に特に役立ちます。

[`mark_preempting`](../../ref/python/run.md#markpreempting) 関数を使用して、中断された sweep runs を自動的に再キューするように W&B を設定します。例えば、以下のコードスニペット

```python
run = wandb.init()  # run を初期化
run.mark_preempting()
```
以下の表は、sweep run の終了ステータスに基づいて W&B が runs をどのように処理するかを示しています。

|Status| Behavior |
|------| ---------|
|Status code 0| run が正常に終了したと見なされ、再キューされません。 |
|Nonzero status| W&B は自動的に run を sweep に関連付けられた run キューに追加します。|
|No status| run が sweep run キューに追加されます。Sweep エージェントは、キューが空になるまで run キューから runs を消費します。キューが空になると、sweep キューは sweep 検索アルゴリズムに基づいて新しい runs の生成を再開します。|