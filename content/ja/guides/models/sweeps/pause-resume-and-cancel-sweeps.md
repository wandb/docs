---
title: スイープを CLI で管理する
description: W&B Sweep を CLI で一時停止、再開、キャンセルします。
menu:
  default:
    identifier: ja-guides-models-sweeps-pause-resume-and-cancel-sweeps
    parent: sweeps
weight: 8
---

W&B Sweep を CLI で一時停止、再開、キャンセルすることができます。W&B Sweep を一時停止すると、新しい W&B Runs を再開するまで実行しないよう W&B エージェントに指示します。Sweep を再開すると、エージェントに新しい W&B Run の実行を続けるよう指示します。W&B Sweep を停止すると、W&B Sweep エージェントに新しい W&B Run の作成や実行を停止するように指示します。W&B Sweep をキャンセルすると、現在実行中の W&B Run を強制終了し、新しい Runs の実行を停止するよう Sweep エージェントに指示します。

それぞれの場合に、W&B Sweep を初期化したときに生成された W&B Sweep ID を指定します。新しいターミナルウィンドウを開いて、以下のコマンドを実行します。新しいターミナルウィンドウを開くと、現在のターミナルウィンドウに W&B Sweep が出力ステートメントを表示している場合でも、コマンドをより簡単に実行できます。

スイープを一時停止、再開、およびキャンセルするための次のガイダンスを使用してください。

### スイープを一時停止する

W&B Sweep を一時的に停止して新しい W&B Run の実行を一時停止します。`wandb sweep --pause` コマンドを使用して W&B Sweep を一時停止します。一時停止したい W&B Sweep ID を指定します。

```bash
wandb sweep --pause entity/project/sweep_ID
```

### スイープを再開する

一時停止している W&B Sweep を `wandb sweep --resume` コマンドで再開します。再開したい W&B Sweep ID を指定します。

```bash
wandb sweep --resume entity/project/sweep_ID
```

### スイープを停止する

W&B スイープを完了し、新しい W&B Runs の実行を停止し、現在実行中の Runs が終了するのを待ちます。

```bash
wandb sweep --stop entity/project/sweep_ID
```

### スイープをキャンセルする

すべての実行中の run を強制終了し、新しい run の実行を停止するためにスイープをキャンセルします。`wandb sweep --cancel` コマンドを使用して W&B Sweep をキャンセルします。キャンセルしたい W&B Sweep ID を指定します。

```bash
wandb sweep --cancel entity/project/sweep_ID
```

CLI コマンドオプションの全リストについては、[wandb sweep]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) CLI リファレンスガイドを参照してください。

### 複数のエージェントにわたってスイープを一時停止、再開、停止、キャンセルする

単一のターミナルから複数のエージェントにわたって W&B Sweep を一時停止、再開、停止、またはキャンセルします。たとえば、マルチコアマシンを持っていると仮定します。W&B Sweep を初期化した後、ターミナルウィンドウを新たに開き、各新しいターミナルに Sweep ID をコピーします。

任意のターミナル内で、`wandb sweep` CLI コマンドを使用して W&B Sweep を一時停止、再開、停止、またはキャンセルします。たとえば、以下のコードスニペットは、CLI を使用して複数のエージェントにわたって W&B Sweep を一時停止する方法を示しています。

```
wandb sweep --pause entity/project/sweep_ID
```

`--resume` フラグと共に Sweep ID を指定して、エージェントにわたって Sweep を再開します。

```
wandb sweep --resume entity/project/sweep_ID
```

W&B エージェントを並列化する方法の詳細については、[Parallelize agents]({{< relref path="./parallelize-agents.md" lang="ja" >}}) を参照してください。