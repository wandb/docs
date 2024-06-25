---
description: W&B Sweep を CLI で一時停止、再開、キャンセルする。
displayed_sidebar: default
---


# Pause, resume, stop and cancel sweeps

<head>
    <title>Pause, resume, stop or cancel W&B Sweeps</title>
</head>

Pause, resume, and cancel a W&B Sweep with the CLI. Pausing a W&B Sweep tells the W&B agent that new W&B Runs should not be executed until the Sweep is resumed. Resuming a Sweep tells the agent to continue executing new W&B Runs. Stopping a W&B Sweep tells the W&B Sweep agent to stop creating or executing new W&B Runs. Cancelling a W&B Sweep tells the Sweep agent to kill currently executing W&B Runs and stop executing new Runs.

各操作では、W&B Sweep を初期化したときに生成された W&B Sweep ID を指定します。オプションとして、新しいターミナルウィンドウを開いて次のコマンドを実行するとよいでしょう。新しいターミナルウィンドウを開くと、現在のターミナルウィンドウで W&B Sweep が出力ステートメントを印刷している場合でも、コマンドを実行しやすくなります。

次のガイドを使用して、sweeps を一時停止、再開、キャンセルします。

### Pause sweeps

新しい W&B Runs の実行を一時停止するために W&B Sweep を一時停止します。`wandb sweep --pause` コマンドを使用して W&B Sweep を一時停止します。一時停止したい W&B Sweep ID を指定します。

```bash
wandb sweep --pause entity/project/sweep_ID
```

### Resume sweeps

一時停止した W&B Sweep を `wandb sweep --resume` コマンドで再開します。再開したい W&B Sweep ID を指定します。

```bash
wandb sweep --resume entity/project/sweep_ID
```

### Stop sweeps

新しい W&B Runs の実行を停止し、現在実行中の Runs を完了させるために W&B sweep を終了します。

```bash
wandb sweep --stop entity/project/sweep_ID
```

### Cancel sweeps

すべての実行中の runs を停止し、新しい runs の実行も停止するために sweep をキャンセルします。`wandb sweep --cancel` コマンドを使用して W&B Sweep をキャンセルします。キャンセルしたい W&B Sweep ID を指定します。

```bash
wandb sweep --cancel entity/project/sweep_ID
```

全 CLI コマンド オプションの一覧については、[wandb sweep](../../ref/cli/wandb-sweep.md) CLI リファレンス ガイドを参照してください。

### Pause, resume, stop, and cancel a sweep across multiple agents

単一のターミナルから複数のエージェントにわたって W&B Sweep を一時停止、再開、停止、またはキャンセルします。たとえば、マルチコアのマシンがあると仮定します。W&B Sweep を初期化した後、新しいターミナルウィンドウを開き、各新しいターミナルに Sweep ID をコピーします。

任意のターミナル内で `wandb sweep` CLI コマンドを使用して、複数のエージェントにわたって W&B Sweep を一時停止、再開、停止、またはキャンセルします。たとえば、次のコードスニペットは、CLI を使用して複数のエージェントにわたって W&B Sweep を一時停止する方法を示しています。

```
wandb sweep --pause entity/project/sweep_ID
```

エージェント全体で Sweep を再開するには、`--resume` フラグと Sweep ID を指定します。

```
wandb sweep --resume entity/project/sweep_ID
```

W&B エージェントの並列化方法の詳細については、[Parallelize agents](./parallelize-agents.md) を参照してください。