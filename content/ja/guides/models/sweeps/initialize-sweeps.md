---
title: sweep を初期化する
description: W&B の Sweep を初期化する
menu:
  default:
    identifier: ja-guides-models-sweeps-initialize-sweeps
    parent: sweeps
weight: 4
---

W&B は、クラウド（standard）やローカル（local、1 台以上のマシンにまたがって）で Sweeps を管理するために _Sweep コントローラ_ を使用します。Run が完了すると、Sweep コントローラが次に実行する新しい Run を記述した一連の指示を発行します。これらの指示は、実際に Run を実行する _エージェント_ が取得します。一般的な W&B Sweep では、コントローラは W&B サーバー上に常駐します。エージェント は _あなたの_ マシンで動作します。

次の コードスニペット は、CLI と Jupyter Notebook または Python スクリプト 内で Sweep を初期化する方法を示します。

{{% alert color="secondary" %}}
1. Sweep を初期化する前に、YAML ファイル、または スクリプト 内の Python の ネストされた 辞書 オブジェクトのいずれかで sweep configuration が定義されていることを確認してください。詳細は、[sweep configuration を定義する]({{< relref path="/guides/models/sweeps/define-sweep-configuration.md" lang="ja" >}}) を参照してください。
2. W&B Sweep と W&B Run は同じ Project に属している必要があります。そのため、W&B を初期化する際（[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}})）に指定する名前は、W&B Sweep を初期化する際（[`wandb.sweep()`]({{< relref path="/ref/python/sdk/functions/sweep.md" lang="ja" >}})）に指定する Project 名と一致していなければなりません。
{{% /alert %}}


{{< tabpane text=true >}}
{{% tab header="Python script or notebook" %}}

W&B SDK を使って Sweep を初期化します。`sweep` パラメータに sweep configuration の 辞書 を渡してください。任意で、W&B Run の出力を保存したい Project 名を Project パラメータ（`project`）として指定できます。Project が指定されていない場合、Run は「Uncategorized」Project に配置されます。

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

[`wandb.sweep()`]({{< relref path="/ref/python/sdk/functions/sweep.md" lang="ja" >}}) 関数は sweep ID を返します。sweep ID には Entity 名と Project 名が含まれます。sweep ID を控えておいてください。

{{% /tab %}}
{{% tab header="CLI" %}}

W&B CLI を使って Sweep を初期化します。設定ファイルの名前を指定してください。任意で、`project` フラグに Project 名を指定できます。Project が指定されていない場合、W&B Run は「Uncategorized」Project に配置されます。

[`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) コマンドを使って Sweep を初期化します。次の コード例 は、`sweeps_demo` Project に対して Sweep を初期化し、設定には `config.yaml` ファイルを使用します。

```bash
wandb sweep --project sweeps_demo config.yaml
```

このコマンドは sweep ID を出力します。sweep ID には Entity 名と Project 名が含まれます。sweep ID を控えておいてください。

{{% /tab %}}
{{< /tabpane >}}