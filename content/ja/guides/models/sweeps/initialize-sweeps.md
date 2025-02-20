---
title: Initialize a sweep
description: W&B Sweep を初期化する
menu:
  default:
    identifier: ja-guides-models-sweeps-initialize-sweeps
    parent: sweeps
weight: 4
---

W&B は _スイープコントローラ_ を使用して、1 台または複数のマシンでクラウド (標準) およびローカル (ローカル) のスイープを管理します。run が完了すると、スイープコントローラは新しい run を実行するための一連の指示を発行します。これらの指示は、実際に run を実行する _エージェント_ が受け取ります。典型的な W&B Sweep では、コントローラは W&B サーバー上に存在し、エージェントは _あなたの_ マシン上に存在します。

以下のコードスニペットは、CLI および Jupyter ノートブックや Python スクリプト内でスイープを初期化する方法を示しています。

{{% alert color="secondary" %}}
1. スイープを初期化する前に、スイープ設定が YAML ファイルまたはスクリプト内のネストされた Python 辞書オブジェクトで定義されていることを確認してください。詳細は、[スイープ設定を定義する]({{< relref path="/guides/models/sweeps/define-sweep-configuration.md" lang="ja" >}})を参照してください。
2. W&B Sweep と W&B Run は同じプロジェクト内になければなりません。そのため、W&B を初期化する際に提供する名前 ([`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}})) は、W&B Sweep を初期化する際に提供するプロジェクトの名前 ([`wandb.sweep`]({{< relref path="/ref/python/sweep.md" lang="ja" >}})) と一致する必要があります。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="Python script or notebook" %}}

W&B SDK を使用してスイープを初期化します。スイープ設定辞書を `sweep` パラメータに渡します。オプションとして、W&B Run の出力を保存したいプロジェクト用のプロジェクトパラメータ (`project`) の名前を提供します。プロジェクトが指定されていない場合、run は「未分類」プロジェクトに配置されます。

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

[`wandb.sweep`]({{< relref path="/ref/python/sweep" lang="ja" >}}) 関数はスイープ ID を返します。スイープ ID にはエンティティ名とプロジェクト名が含まれます。スイープ ID をメモしておきましょう。

{{% /tab %}}
{{% tab header="CLI" %}}

W&B CLI を使用してスイープを初期化します。設定ファイルの名前を提供します。オプションとして、`project` フラグ用のプロジェクト名を提供します。プロジェクトが指定されていない場合、W&B Run は「未分類」プロジェクトに配置されます。

[`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) コマンドを使用してスイープを初期化します。次のコード例は `sweeps_demo` プロジェクトに対してスイープを初期化し、設定に `config.yaml` ファイルを使用します。

```bash
wandb sweep --project sweeps_demo config.yaml
```

このコマンドはスイープ ID を出力します。スイープ ID にはエンティティ名とプロジェクト名が含まれます。スイープ ID をメモしておきましょう。

{{% /tab %}}
{{< /tabpane >}}