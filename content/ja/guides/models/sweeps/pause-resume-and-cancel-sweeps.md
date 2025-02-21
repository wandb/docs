---
title: Manage sweeps with the CLI
description: CLI で W&B Sweep を一時停止、再開、およびキャンセルします。
menu:
  default:
    identifier: ja-guides-models-sweeps-pause-resume-and-cancel-sweeps
    parent: sweeps
weight: 8
---

CLI で W&B Sweep を一時停止、再開、およびキャンセルします。W&B Sweep を一時停止すると、Sweep が再開されるまで新しい W&B Run を実行しないように W&B エージェント に指示します。Sweep を再開すると、新しい W&B Run の実行を続行するようにエージェントに指示します。W&B Sweep を停止すると、新しい W&B Run の作成または実行を停止するように W&B Sweep エージェント に指示します。W&B Sweep をキャンセルすると、現在実行中の W&B Run を強制終了し、新しい Run の実行を停止するように Sweep エージェント に指示します。

いずれの場合も、W&B Sweep の初期化時に生成された W&B Sweep ID を指定してください。必要に応じて、新しい ターミナル ウィンドウを開いて、次の コマンド を実行します。W&B Sweep が現在の ターミナル ウィンドウに出力ステートメントを出力している場合、新しい ターミナル ウィンドウを使用すると、 コマンド を簡単に実行できます。

次の ガイド を使用して、sweep を一時停止、再開、およびキャンセルします。

### sweep の一時停止

新しい W&B Run の実行を一時的に停止するように W&B Sweep を一時停止します。W&B Sweep を一時停止するには、`wandb sweep --pause` コマンド を使用します。一時停止する W&B Sweep ID を指定します。

```bash
wandb sweep --pause entity/project/sweep_ID
```

### sweep の再開

`wandb sweep --resume` コマンド で、一時停止した W&B Sweep を再開します。再開する W&B Sweep ID を指定します。

```bash
wandb sweep --resume entity/project/sweep_ID
```

### sweep の停止

新しい W&B Run の実行を停止し、現在実行中の Run を終了させるには、W&B sweep を終了します。

```bash
wandb sweep --stop entity/project/sweep_ID
```

### sweep のキャンセル

実行中のすべての run を強制終了し、新しい run の実行を停止するには、sweep をキャンセルします。W&B Sweep をキャンセルするには、`wandb sweep --cancel` コマンド を使用します。キャンセルする W&B Sweep ID を指定します。

```bash
wandb sweep --cancel entity/project/sweep_ID
```

CLI コマンド オプションの完全なリストについては、[wandb sweep]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) CLI リファレンス ガイド を参照してください。

### 複数の エージェント にわたる sweep の一時停止、再開、停止、およびキャンセル

単一の ターミナル から、複数の エージェント にわたる W&B Sweep を一時停止、再開、停止、またはキャンセルします。たとえば、マルチコア マシンがあるとします。W&B Sweep を初期化したら、新しい ターミナル ウィンドウを開き、各新しい ターミナル に Sweep ID をコピーします。

任意の ターミナル 内で、`wandb sweep` CLI コマンド を使用して、W&B Sweep を一時停止、再開、停止、またはキャンセルします。たとえば、次の コードスニペット は、CLI で複数の エージェント にわたる W&B Sweep を一時停止する方法を示しています。

```
wandb sweep --pause entity/project/sweep_ID
```

エージェント 全体で Sweep を再開するには、Sweep ID とともに `--resume` フラグを指定します。

```
wandb sweep --resume entity/project/sweep_ID
```

W&B エージェント を並列化する方法の詳細については、[エージェント の並列化]({{< relref path="./parallelize-agents.md" lang="ja" >}})を参照してください。
