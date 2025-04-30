---
title: run を再開
description: 一時停止または終了した W&B Run を再開する
menu:
  default:
    identifier: ja-guides-models-track-runs-resuming
    parent: what-are-runs
---

実行が停止またはクラッシュした場合にどのように動作するべきかを指定します。実行を再開または自動的に再開するためには、その実行に関連付けられた一意の実行IDを`id`パラメータとして指定する必要があります。

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```

{{% alert %}}
W&Bは、実行を保存したいW&B Projectの名前を指定することをお勧めします。
{{% /alert %}}

W&Bがどのように対応するかを決定するために、次の引数の1つを`resume`パラメータに渡します。各場合において、W&Bは最初に実行IDが既に存在するかを確認します。

|引数 | 説明 | 実行IDが存在する場合 | 実行IDが存在しない場合 | ユースケース |
| --- | --- | -- | --| -- |
| `"must"` | W&Bは実行IDで指定された実行を再開する必要があります。 | 同じ実行IDで実行を再開します。 | W&Bがエラーを発生させます。 | 同じ実行IDを使用する必要がある実行を再開します。 |
| `"allow"`| 実行IDが存在する場合、W&Bが実行を再開することを許可します。 | 同じ実行IDで実行を再開します。 | 指定された実行IDで新しい実行を初期化します。 | 既存の実行を上書きせずに実行を再開します。 |
| `"never"`| 実行IDで指定された実行をW&Bが再開することを許可しない。 | W&Bがエラーを発生させます。 | 指定された実行IDで新しい実行を初期化します。 | |

`resume="auto"`を指定することで、W&Bが自動的にあなたの代わりに実行を再開しようとします。ただし、同じディレクトリーから実行を再開することを保証する必要があります。詳細は、[実行を自動的に再開する設定を有効にする]({{< relref path="#enable-runs-to-automatically-resume" lang="ja" >}})セクションを参照してください。

以下の例では、`<>`で囲まれた値をあなた自身のものに置き換えてください。

## 同じ実行IDを使用して実行を再開する
実行が停止、クラッシュ、または失敗した場合、同じ実行IDを使用して再開できます。これを行うには、実行を初期化し、以下を指定します。

* `resume`パラメータを`"must"`に設定します（`resume="must"`）。
* 停止またはクラッシュした実行の実行IDを指定します。

以下のコードスニペットは、W&B Python SDKでこれを達成する方法を示しています。

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="must")
```

{{% alert color="secondary" %}}
複数のプロセスが同じ`id`を同時に使用すると予期しない結果が発生します。

複数プロセスの管理方法については、[分散トレーニング実験のログ]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}})を参照してください。
{{% /alert %}}

## 既存の実行を上書きせずに実行を再開する
停止またはクラッシュした実行を、既存の実行を上書きせずに再開します。これは、プロセスが正常に終了しない場合に特に役立ちます。次回W&Bを開始すると、W&Bは最後のステップからログを開始します。

W&Bで実行を初期化する場合、`resume`パラメータを`"allow"`(`resume="allow"`)で設定します。停止またはクラッシュした実行の実行IDを指定します。以下のコードスニペットは、W&B Python SDKでこれを達成する方法を示しています。

```python
import wandb

run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="allow")
```

## 実行を自動的に再開するように設定を有効にする
以下のコードスニペットは、Python SDKまたは環境変数を使用して実行を自動的に再開する方法を示します。

{{< tabpane text=true >}}
  {{% tab header="W&B Python SDK" %}}
以下のコードスニペットは、Python SDKでW&B実行IDを指定する方法を示しています。

`<>`で囲まれた値をあなた自身のものに置き換えてください。

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```  
  {{% /tab %}}
  {{% tab header="シェルスクリプト" %}}

次の例は、bashスクリプトでW&B `WANDB_RUN_ID`変数を指定する方法を示しています。

```bash title="run_experiment.sh"
RUN_ID="$1"

WANDB_RESUME=allow WANDB_RUN_ID="$RUN_ID" python eval.py
```

ターミナル内で、W&B実行IDと共にシェルスクリプトを実行できます。次のコードスニペットは実行ID `akj172`を渡します。

```bash
sh run_experiment.sh akj172 
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert color="secondary" %}}
自動再開は、プロセスが失敗したプロセスと同じファイルシステムの上で再起動された場合にのみ機能します。
{{% /alert %}}

例えば、`train.py`というPythonスクリプトを`Users/AwesomeEmployee/Desktop/ImageClassify/training/`というディレクトリーで実行するとします。`train.py`内で、自動再開を有効にする実行が作成されます。次にトレーニングスクリプトが停止されたとします。この実行を再開するには、`Users/AwesomeEmployee/Desktop/ImageClassify/training/`内で`train.py`スクリプトを再起動する必要があります。

{{% alert %}}
ファイルシステムを共有できない場合は、`WANDB_RUN_ID`環境変数を指定するか、W&B Python SDKで実行IDを渡します。実行IDの詳細は、"What are runs?"ページの[カスタム実行ID]({{< relref path="./#custom-run-ids" lang="ja" >}})セクションを参照してください。
{{% /alert %}}

## 中断可能なSweepsの実行を再開する
中断されたスイープ実行を自動的に再キューします。これは、スイープエージェントを停止の対象となる計算環境（SLURMジョブの中断可能なキュー、EC2スポットインスタンス、Google Cloud中断可能VMなど）で実行する場合に特に役立ちます。

[`mark_preempting`]({{< relref path="/ref/python/run.md#mark_preempting" lang="ja" >}})関数を使用して、W&Bが中断されたスイープ実行を自動的に再キューできるようにします。以下のコードスニペットの例

```python
run = wandb.init()  # 実行を初期化
run.mark_preempting()
```

以下の表は、スイープ実行の終了ステータスに基づいてW&Bが実行をどのように扱うかを概説しています。

|ステータス| 振る舞い |
|------| ---------|
|ステータスコード0| 実行は成功裏に終了したと見なされ、再キューされません。 |
|非ゼロステータス| W&Bは自動的に実行をスイープに関連付けられたランキューに追加します。|
|ステータスなし| 実行はスイープランキューに追加されます。スイープエージェントは、キューが空になるまでランキューから実行を消費します。キューが空になると、スイープキューはスイープ検索アルゴリズムに基づいて新しい実行の生成を再開します。|