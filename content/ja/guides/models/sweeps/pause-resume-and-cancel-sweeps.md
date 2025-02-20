---
title: Manage sweeps with the CLI
description: CLI を使用して W&B Sweep を一時停止、再開、およびキャンセルします。
menu:
  default:
    identifier: ja-guides-models-sweeps-pause-resume-and-cancel-sweeps
    parent: sweeps
weight: 8
---

W&B Sweep を CLI で一時停止、再開、およびキャンセルします。W&B Sweep を一時停止すると、新しい W&B Runs が Sweep を再開するまで起動されないことを W&B エージェントに伝えます。Sweep を再開すると、エージェントに新しい W&B Runs の実行を続行するように指示します。W&B Sweep を停止すると、W&B Sweep エージェントに新しい W&B Runs の作成または実行を停止するよう指示します。W&B Sweep をキャンセルすると、Sweep エージェントに現在実行中の W&B Runs を中止し、新しい Runs の実行を停止するよう指示します。

それぞれの場合で、W&B Sweep を初期化したときに生成された W&B Sweep ID を提供します。オプションとして、新しいターミナルウィンドウを開いて、後続のコマンドを実行することができます。新しいターミナルウィンドウを使用すれば、現在のターミナルウィンドウに W&B Sweep が出力文を表示していても、コマンドを簡単に実行できます。

以下のガイダンスを使用して、スイープを一時停止、再開、キャンセルします。

### スイープの一時停止

W&B Sweep を一時停止して、新しい W&B Runs の実行を一時的に停止します。`wandb sweep --pause` コマンドを使用して、W&B Sweep を一時停止します。一時停止したい W&B Sweep ID を指定してください。

```bash
wandb sweep --pause entity/project/sweep_ID
```

### スイープの再開

一時停止された W&B Sweep を `wandb sweep --resume` コマンドで再開します。再開したい W&B Sweep ID を指定してください。

```bash
wandb sweep --resume entity/project/sweep_ID
```

### スイープの停止

新しい W&B Runs の実行を停止し、現在実行中の Runs が終了するように、W&B Sweep を終了します。

```bash
wandb sweep --stop entity/project/sweep_ID
```

### スイープのキャンセル

すべての実行中の runs を中止し、新しい runs の実行を停止するために sweep をキャンセルします。`wandb sweep --cancel` コマンドを使用して、W&B Sweep をキャンセルします。キャンセルしたい W&B Sweep ID を指定してください。

```bash
wandb sweep --cancel entity/project/sweep_ID
```

CLI コマンドオプションの全リストについては、[wandb sweep]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) CLI リファレンスガイドを参照してください。

### 複数のエージェントにまたがるスイープの一時停止、再開、停止およびキャンセル

単一のターミナルから複数のエージェントにまたがる W&B Sweep を一時停止、再開、停止またはキャンセルします。たとえば、マルチコアマシンを持っているとします。W&B Sweep を初期化した後、新しいターミナルウィンドウを開き、それぞれの新しいターミナルに Sweep ID をコピーします。

いずれのターミナルでも、`wandb sweep` CLI コマンドを使用して、W&B Sweep を一時停止、再開、停止またはキャンセルします。たとえば、次のコードスニペットは、CLI で複数のエージェントにわたる W&B Sweep を一時停止する方法を示しています。

```
wandb sweep --pause entity/project/sweep_ID
```

Sweep ID とともに `--resume` フラグを指定して、エージェント全体で Sweep を再開します。

```
wandb sweep --resume entity/project/sweep_ID
```

W&B エージェントを並列化する方法の詳細については、[エージェントの並列化]({{< relref path="./parallelize-agents.md" lang="ja" >}})を参照してください。