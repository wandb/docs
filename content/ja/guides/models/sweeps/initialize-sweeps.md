---
title: sweep を初期化する
description: W&B で Sweep を初期化する
menu:
  default:
    identifier: ja-guides-models-sweeps-initialize-sweeps
    parent: sweeps
weight: 4
---

W&B は、_Sweep Controller_ を使用して、クラウド (標準)、ローカル (ローカル) の 1 台以上のマシンで スイープを管理します。run が完了すると、sweep controller は新しい run を実行するための新しい指示を発行します。これらの指示は、実際に run を実行する _agents_ によって受け取られます。典型的な W&B Sweep では、controller は W&B サーバー上に存在し、agents は_あなたの_マシン上に存在します。

以下のコードスニペットは、CLI、および Jupyter Notebook や Python スクリプト内でスイープを初期化する方法を示しています。

{{% alert color="secondary" %}}
1. スイープを初期化する前に、スイープ設定が YAML ファイルまたはスクリプト内のネストされた Python 辞書オブジェクトで定義されていることを確認してください。詳細については、[sweep configuration を定義する]({{< relref path="/guides/models/sweeps/define-sweep-configuration.md" lang="ja" >}}) を参照してください。
2. W&B Sweep と W&B Run は同じ Project 内にある必要があります。そのため、W&B を初期化するときに指定する名前 ([`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}})) は、W&B Sweep を初期化するときに提供する Project 名 ([`wandb.sweep`]({{< relref path="/ref/python/sweep.md" lang="ja" >}})) と一致している必要があります。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="Python script or notebook" %}}

W&B SDK を使用してスイープを初期化します。スイープ設定辞書を `sweep` パラメータに渡します。オプションで、W&B Run の出力を保存したい Project の名前を Project パラメータ (`project`) に指定することができます。Project が指定されていない場合は、run は「Uncategorized」Project に置かれます。

```python
import wandb

# スイープ設定の例
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

[`wandb.sweep`]({{< relref path="/ref/python/sweep" lang="ja" >}}) 関数はスイープ ID を返します。スイープ ID には entity 名と Project 名が含まれます。スイープ ID をメモしておいてください。

{{% /tab %}}
{{% tab header="CLI" %}}

W&B CLI を使用してスイープを初期化します。設定ファイルの名前を指定します。オプションで、`project` フラグに Project の名前を指定することができます。Project が指定されていない場合、W&B Run は「Uncategorized」Project に配置されます。

[`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) コマンドを使用してスイープを初期化します。次のコード例は、`sweeps_demo` Project のスイープを初期化し、`config.yaml` ファイルを設定に使用しています。

```bash
wandb sweep --project sweeps_demo config.yaml
```

このコマンドはスイープ ID を出力します。スイープ ID には entity 名と Project 名が含まれます。スイープ ID をメモしておいてください。

{{% /tab %}}
{{< /tabpane >}}