---
title: Initialize a sweep
description: W&B Sweep の初期化
menu:
  default:
    identifier: ja-guides-models-sweeps-initialize-sweeps
    parent: sweeps
weight: 4
---

W&B は、クラウド（標準）、ローカル（ローカル）の 1 台以上のマシンで Sweeps を管理するために、 _Sweep Controller_ を使用します。run が完了すると、sweep controller は、実行する新しい run を記述した新しい一連の指示を発行します。これらの指示は、実際に run を実行する _エージェント_ によって取得されます。典型的な W&B Sweep では、コントローラは W&B サーバー上に存在します。エージェントは _あなたの_ マシン上に存在します。

以下のコードスニペットは、CLI および Jupyter Notebook または Python スクリプト内で Sweeps を初期化する方法を示しています。

{{% alert color="secondary" %}}
1. sweep を初期化する前に、sweep configuration が YAML ファイルまたはスクリプト内のネストされた Python 辞書 object で定義されていることを確認してください。詳細については、[sweep configuration の定義]({{< relref path="/guides/models/sweeps/define-sweep-configuration.md" lang="ja" >}})を参照してください。
2. W&B Sweep と W&B Run は、同じ project 内にある必要があります。したがって、W&B を初期化するときに指定する名前 ([`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}})) は、W&B Sweep を初期化するときに指定する project の名前 ([`wandb.sweep`]({{< relref path="/ref/python/sweep.md" lang="ja" >}})) と一致する必要があります。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="Python script or notebook" %}}

W&B SDK を使用して sweep を初期化します。sweep configuration 辞書を `sweep` パラメータに渡します。オプションで、W&B Run の出力を保存する project パラメータ（`project`）の project 名を指定します。project が指定されていない場合、run は「未分類」の project に配置されます。

```python
import wandb

# sweep configuration の例
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

[`wandb.sweep`]({{< relref path="/ref/python/sweep" lang="ja" >}}) 関数は、sweep ID を返します。sweep ID には、Entity 名と project 名が含まれます。sweep ID をメモしておいてください。

{{% /tab %}}
{{% tab header="CLI" %}}

W&B CLI を使用して sweep を初期化します。configuration ファイルの名前を指定します。オプションで、`project` フラグの project 名を指定します。project が指定されていない場合、W&B Run は「未分類」の project に配置されます。

[`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) コマンドを使用して、sweep を初期化します。次のコード例では、`sweeps_demo` project の sweep を初期化し、configuration に `config.yaml` ファイルを使用します。

```bash
wandb sweep --project sweeps_demo config.yaml
```

このコマンドは、sweep ID を出力します。sweep ID には、Entity 名と project 名が含まれます。sweep ID をメモしておいてください。

{{% /tab %}}
{{< /tabpane >}}
