---
title: CLI でスイープを管理する
description: CLI を使って W&B Sweep の一時停止、再開、キャンセルを行う方法をご紹介します。
menu:
  default:
    identifier: pause-resume-and-cancel-sweeps
    parent: sweeps
weight: 8
---

CLI を使って W&B Sweep を一時停止、再開、キャンセルする方法をご紹介します。W&B Sweep を一時停止すると、エージェントは新しい W&B Run を Sweep が再開されるまで実行しません。Sweep を再開すると、エージェントは新しい W&B Run の実行を再開します。W&B Sweep を停止すると、Sweep エージェントは新しい W&B Run の作成や実行を停止します。W&B Sweep をキャンセルすると、Sweep エージェントは現在実行中の W&B Run を強制終了し、新しい Run の実行も止めます。

いずれの場合も、W&B Sweep を初期化したときに生成された W&B Sweep ID を指定してください。必要に応じて、新しいターミナルウィンドウを開いてコマンドを実行すると便利です。現在のターミナルに出力が表示されている場合でも、別のウィンドウでコマンドが実行しやすくなります。

以下の手順に従って、Sweep の一時停止、再開、キャンセルを行ってください。

### Sweep を一時停止する

W&B Sweep を一時停止すると、新しい W&B Run の実行が一時的にストップします。Sweep を一時停止するには、`wandb sweep --pause` コマンドを使用します。一時停止したい W&B Sweep の ID を指定してください。

```bash
wandb sweep --pause entity/project/sweep_ID
```

### Sweep を再開する

一時停止した W&B Sweep を再開するには、`wandb sweep --resume` コマンドを使います。再開したい W&B Sweep の ID を指定してください。

```bash
wandb sweep --resume entity/project/sweep_ID
```

### Sweep を停止する

W&B Sweep を終了して新しい W&B Run の実行を止め、現在実行中の Run が完了するまで待ちます。

```bash
wandb sweep --stop entity/project/sweep_ID
```

### Sweep をキャンセルする

実行中の Run をすべて強制終了し、新しい Run も止めたい場合は Sweep をキャンセルできます。Sweep をキャンセルするには、`wandb sweep --cancel` コマンドを使用します。キャンセルしたい W&B Sweep の ID を指定してください。

```bash
wandb sweep --cancel entity/project/sweep_ID
```

CLI コマンドの全オプション一覧については、[wandb sweep]({{< relref "/ref/cli/wandb-sweep.md" >}}) CLI リファレンスガイドを参照してください。

### 複数のエージェントで Sweep の一時停止・再開・停止・キャンセルを行う

1 つのターミナルから複数のエージェントで W&B Sweep の一時停止、再開、停止、キャンセルができます。たとえばマルチコアマシンをお使いの場合、W&B Sweep を初期化した後、複数のターミナルウィンドウを開き、それぞれに Sweep ID をコピーします。

どのターミナルからでも、`wandb sweep` CLI コマンドを使って Sweep の一時停止、再開、停止、キャンセルが行えます。次のコードスニペットは CLI を使って複数エージェントに対して Sweep を一時停止する例です。

```
wandb sweep --pause entity/project/sweep_ID
```

Sweep を複数のエージェントで再開する場合は、`--resume` フラグと Sweep ID を指定します。

```
wandb sweep --resume entity/project/sweep_ID
```

W&B エージェントの並列化について詳しくは、[Parallelize agents]({{< relref "./parallelize-agents.md" >}}) をご覧ください。