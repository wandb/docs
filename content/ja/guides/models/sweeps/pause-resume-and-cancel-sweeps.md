---
title: CLI で Sweeps を管理する
description: CLI で W&B Sweep を一時停止、再開、キャンセルする。
menu:
  default:
    identifier: ja-guides-models-sweeps-pause-resume-and-cancel-sweeps
    parent: sweeps
weight: 8
---

CLI で W&B Sweep を一時停止、再開、停止、キャンセルできます。W&B Sweep を一時停止すると、W&B エージェントに対して Sweep を再開するまで新しい W&B Runs を実行しないよう指示します。Sweep を再開すると、エージェントに新しい W&B Runs の実行を続けるよう指示します。W&B Sweep を停止すると、W&B Sweep agent に新しい W&B Runs の作成や実行をやめるよう指示します。W&B Sweep をキャンセルすると、Sweep agent に現在実行中の W&B Runs を強制終了し、新しい Runs の実行を停止するよう指示します。

いずれの場合も、W&B Sweep を初期化したときに生成された W&B Sweep ID を指定します。必要に応じて、別のターミナルウィンドウを開いて以下のコマンドを実行してください。現在のターミナルウィンドウで W&B Sweep が出力を表示している場合でも、別ウィンドウならコマンドを実行しやすくなります。

Sweeps を一時停止、再開、キャンセルするには、以下を参照してください。

### Sweeps を一時停止

W&B Sweep を一時停止すると、新しい W&B Runs の実行を一時的に停止します。W&B Sweep を一時停止するには、`wandb sweep --pause` コマンドを使用します。一時停止したい W&B Sweep の ID を指定します。

```bash
wandb sweep --pause entity/project/sweep_ID
```

### Sweeps を再開

一時停止中の W&B Sweep は `wandb sweep --resume` コマンドで再開できます。再開したい W&B Sweep の ID を指定します:

```bash
wandb sweep --resume entity/project/sweep_ID
```

### Sweeps を停止

W&B Sweep を終了すると、新しい W&B Runs の実行を停止し、現在実行中の Runs の終了を待ちます。

```bash
wandb sweep --stop entity/project/sweep_ID
```

### Sweeps をキャンセル

Sweep をキャンセルすると、実行中の Runs をすべて強制終了し、新しい Runs の実行を停止します。W&B Sweep をキャンセルするには、`wandb sweep --cancel` コマンドを使用します。キャンセルしたい W&B Sweep の ID を指定します。

```bash
wandb sweep --cancel entity/project/sweep_ID
```

CLI コマンドオプションの一覧は、[wandb sweep]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) の CLI リファレンスガイドを参照してください。

### 複数のエージェントにまたがって Sweep を一時停止、再開、停止、キャンセルする

単一のターミナルから、複数のエージェントにまたがる W&B Sweep を一時停止、再開、停止、キャンセルできます。たとえばマルチコアのマシンを使っているとします。W&B Sweep を初期化した後、新しいターミナルウィンドウを開き、各ターミナルに Sweep ID をコピーします。

任意のターミナルで、`wandb sweep` CLI コマンドを使って W&B Sweep を一時停止、再開、停止、またはキャンセルします。たとえば、次の コードスニペット は CLI を使って複数のエージェントにまたがる W&B Sweep を一時停止する方法を示しています:

```
wandb sweep --pause entity/project/sweep_ID
```

エージェント全体で Sweep を再開するには、Sweep ID と一緒に `--resume` フラグを指定します:

```
wandb sweep --resume entity/project/sweep_ID
```

W&B エージェントの並列化方法の詳細は、[エージェントを並列化する]({{< relref path="./parallelize-agents.md" lang="ja" >}}) を参照してください。