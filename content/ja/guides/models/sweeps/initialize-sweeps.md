---
title: sweep を初期化する
description: W&B Sweep を初期化する
menu:
  default:
    identifier: initialize-sweeps
    parent: sweeps
weight: 4
---

W&B では、_Sweep Controller_ を使用して、クラウド（standard）やローカル（local）の複数マシン上で sweeps を管理します。run が完了すると、sweep controller は新たな run を実行するための指示を出します。これらの指示は、実際に run を実行する _agent_ によって受け取られます。一般的な W&B Sweep では、controller は W&B サーバー上に存在し、agent は _あなた_ のマシン上で動作します。

以下のコードスニペットは、CLI や Jupyter Notebook、Python スクリプトで sweep を初期化する方法を示しています。

{{% alert color="secondary" %}}
1. sweep を初期化する前に、YAML ファイルまたはスクリプト内のネストされた Python 辞書オブジェクトのいずれかで sweep configuration が定義されていることを確認してください。詳細は [Define sweep configuration]({{< relref "/guides/models/sweeps/define-sweep-configuration.md" >}}) をご覧ください。
2. W&B Sweep と W&B Run は必ず同じ Project 内で実行してください。そのため、W&B を初期化する際に指定する名前（[`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}})）は、W&B Sweep を初期化する際に指定する Project 名（[`wandb.sweep()`]({{< relref "/ref/python/sdk/functions/sweep.md" >}})）と一致させる必要があります。
{{% /alert %}}


{{< tabpane text=true >}}
{{% tab header="Python スクリプトまたはノートブック" %}}

W&B SDK を使って sweep を初期化します。sweep configuration の辞書を `sweep` パラメータに渡してください。オプションで Project の名前を `project` パラメータとして指定できます。Project を指定しない場合、run は「Uncategorized」プロジェクトに保存されます。

```python
import wandb

# sweep 設定例
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

[`wandb.sweep()`]({{< relref "/ref/python/sdk/functions/sweep.md" >}}) 関数は sweep ID を返します。sweep ID には entity 名と project 名が含まれます。sweep ID をメモしておいてください。

{{% /tab %}}
{{% tab header="CLI" %}}

W&B CLI を使用して sweep を初期化します。設定ファイル名を指定し、オプションで Project の名前を `project` フラグで指定可能です。Project を指定しない場合、W&B Run は「Uncategorized」プロジェクトに保存されます。

[`wandb sweep`]({{< relref "/ref/cli/wandb-sweep.md" >}}) コマンドを使って sweep を初期化します。以下のコード例では、`sweeps_demo` プロジェクトで `config.yaml` ファイルを使用して sweep を初期化しています。

```bash
wandb sweep --project sweeps_demo config.yaml
```

このコマンドを実行すると、sweep ID が出力されます。sweep ID には entity 名と project 名が含まれます。sweep ID を必ずメモしてください。

{{% /tab %}}
{{< /tabpane >}}