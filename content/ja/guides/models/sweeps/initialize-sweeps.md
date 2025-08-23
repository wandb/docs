---
title: sweep を初期化する
description: W&B Sweep を初期化する
menu:
  default:
    identifier: ja-guides-models-sweeps-initialize-sweeps
    parent: sweeps
weight: 4
---

W&B は、クラウド（standard）、ローカル（local）で 1 台または複数台のマシンにわたって Sweeps を管理するために _Sweep Controller_ を使用します。run が完了すると、sweep controller が新しい run を実行するための指示を発行します。これらの指示は、実際に run を実行する _agent_ によって受け取られます。一般的な W&B Sweep では、controller は W&B サーバー上に存在し、agent は _あなた自身の_ マシン上で動作します。

以下のコードスニペットは、CLI や Jupyter Notebook、Python script で sweep を初期化する方法を示しています。

{{% alert color="secondary" %}}
1. sweep を初期化する前に、YAML ファイルまたは script 内のネストされた Python 辞書オブジェクトとして sweep configuration が定義されていることを確認してください。詳しくは、[Sweep configuration の定義方法]({{< relref path="/guides/models/sweeps/define-sweep-configuration.md" lang="ja" >}})をご覧ください。
2. W&B Sweep と W&B Run は同じプロジェクト内である必要があります。そのため、W&B の初期化時（[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}})）に指定するプロジェクト名は、W&B Sweep の初期化時（[`wandb.sweep()`]({{< relref path="/ref/python/sdk/functions/sweep.md" lang="ja" >}})）と一致させてください。
{{% /alert %}}


{{< tabpane text=true >}}
{{% tab header="Python script または notebook" %}}

W&B SDK を使って sweep を初期化します。sweep configuration の辞書を `sweep` パラメータとして渡してください。さらに、出力先となるプロジェクト名を `project` パラメータで指定できます。プロジェクトが指定されていない場合、run の出力は「Uncategorized」プロジェクトに保存されます。

```python
import wandb

# Sweep 設定例
sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [5, 10, 15]},
        "lr": {"max": 0.1, "min": 0.0001},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="project-name")
```

[`wandb.sweep()`]({{< relref path="/ref/python/sdk/functions/sweep.md" lang="ja" >}}) 関数は sweep ID を返します。この sweep ID には entity 名と project 名が含まれます。sweep ID を控えておいてください。

{{% /tab %}}
{{% tab header="CLI" %}}

W&B CLI を使って sweep を初期化します。設定ファイルの名前を指定してください。必要であれば、`project` フラグでプロジェクト名も指定できます。プロジェクトが指定されていない場合、W&B Run は「Uncategorized」プロジェクトに保存されます。

[`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) コマンドを使って sweep を初期化します。次のコード例は、`sweeps_demo` プロジェクト用に sweep を初期化し、`config.yaml` ファイルを設定として使用しています。

```bash
wandb sweep --project sweeps_demo config.yaml
```

このコマンドを実行すると sweep ID が表示されます。sweep ID には entity 名と project 名が含まれています。sweep ID を控えておいてください。

{{% /tab %}}
{{< /tabpane >}}