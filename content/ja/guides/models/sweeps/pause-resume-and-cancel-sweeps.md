---
title: CLI で sweep を管理する
description: CLI を使って W&B Sweep の一時停止、再開、およびキャンセルを行う方法。
menu:
  default:
    identifier: ja-guides-models-sweeps-pause-resume-and-cancel-sweeps
    parent: sweeps
weight: 8
---

CLI を使って W&B Sweep を一時停止、再開、キャンセルできます。W&B Sweep を一時停止すると、W&B エージェントに新しい W&B Run の実行を一時的に停止するよう指示します。Sweep を再開すると、エージェントは新しい W&B Run の実行を再開します。W&B Sweep を停止すると、W&B Sweep agent に新規 W&B Run の生成や実行の停止を指示します。W&B Sweep をキャンセルすると、Sweep agent は現在実行中の W&B Run を強制終了し、新しい Run の実行も停止します。

いずれの場合も、W&B Sweep の初期化時に生成された Sweep ID を指定してください。続くコマンドを実行する際は、必要に応じて新しいターミナルウィンドウを開くとよいでしょう。W&B Sweep から出力が多い場合、新しいターミナルを使うことでコマンドの実行がしやすくなります。

以下の手順に従って、スイープの一時停止、再開、キャンセルを行いましょう。

### スイープを一時停止する

W&B Sweep を一時的に新しい W&B Run の実行を止めたい場合は、`wandb sweep --pause` コマンドを使います。停止したい W&B Sweep の ID を指定してください。

```bash
wandb sweep --pause entity/project/sweep_ID
```

### スイープを再開する

一時停止した W&B Sweep は `wandb sweep --resume` コマンドで再開できます。再開したい W&B Sweep の ID を指定してください。

```bash
wandb sweep --resume entity/project/sweep_ID
```

### スイープを停止する

新しい W&B Run の実行を止め、現在実行中の Run のみ最後まで動作させたい場合は、スイープを停止します。

```bash
wandb sweep --stop entity/project/sweep_ID
```

### スイープをキャンセルする

実行中の全 Run を強制終了し、新たな Run の実行も止めるには、`wandb sweep --cancel` コマンドを使用して W&B Sweep をキャンセルします。キャンセルしたい W&B Sweep の ID を指定してください。

```bash
wandb sweep --cancel entity/project/sweep_ID
```

CLI コマンドのオプション一覧については、[wandb sweep]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) CLI リファレンスガイドをご覧ください。

### 複数エージェントでのスイープの一時停止・再開・停止・キャンセル

複数エージェントで動作している W&B Sweep を、ひとつのターミナルから一時停止、再開、停止、キャンセルすることができます。例えばマルチコアマシンを使っている場合、W&B Sweep を初期化した後、各ターミナルウィンドウに Sweep ID をコピーしてエージェントを起動します。

どのターミナルからでも `wandb sweep` CLI コマンドで W&B Sweep を一時停止・再開・停止・キャンセルできます。例えば以下のコードスニペットは、CLI を使って複数エージェントで動作中の W&B Sweep を一時停止する例です。

```
wandb sweep --pause entity/project/sweep_ID
```

スイープ全体を再開したい場合は、`--resume` フラグと Sweep ID を指定してください。

```
wandb sweep --resume entity/project/sweep_ID
```

W&B エージェントの並列化について詳しくは、[エージェントの並列化]({{< relref path="./parallelize-agents.md" lang="ja" >}}) をご参照ください。