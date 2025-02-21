---
title: Kubeflow Pipelines (kfp)
description: W&B を Kubeflow パイプライン と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-kubeflow-pipelines-kfp
    parent: integrations
weight: 170
---

[Kubeflow Pipelines (kfp) ](https://www.kubeflow.org/docs/components/pipelines/overview/)は、Docker コンテナに基づいたポータブルでスケーラブルな機械学習 (ML) ワークフローを構築およびデプロイするためのプラットフォームです。

このインテグレーションにより、ユーザーは kfp Python 関数コンポーネントにデコレータを適用して、パラメータおよび Artifacts を自動的に W&B にログすることができます。

この機能は `wandb==0.12.11`で有効になり、`kfp<2.0.0`が必要です。

## サインアップして APIキーを作成

API キーは、あなたのマシンを W&B に認証します。API キーはあなたのユーザープロファイルから生成することができます。

{{% alert %}}
よりスムーズな方法として、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして API キーを生成することができます。表示された API キーをコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロファイルアイコンをクリックします。
1. **User Settings**を選択し、**API Keys**セクションまでスクロールします。
1. **Reveal**をクリックします。表示された API キーをコピーします。API キーを隠すには、ページをリロードします。

## `wandb`ライブラリをインストールしてログイン

ローカルに`wandb`ライブラリをインストールし、ログインします。

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})をあなたの API キーに設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb`ライブラリをインストールしてログインします。

    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## コンポーネントをデコレート

`@wandb_log`デコレータを追加し、通常通りにコンポーネントを作成します。これにより、パイプラインを実行するたびに入出力パラメータと Artifacts が W&B に自動的にログされます。

```python
from kfp import components
from wandb.integration.kfp import wandb_log


@wandb_log
def add(a: float, b: float) -> float:
    return a + b


add = components.create_component_from_func(add)
```

## 環境変数をコンテナに渡す

環境変数をコンテナに明示的に渡す必要がある場合があります。双方向リンクを設定するには、環境変数 `WANDB_KUBEFLOW_URL` を Kubeflow Pipelines インスタンスの基本 URL に設定する必要があります。例えば、`https://kubeflow.mysite.com` などです。

```python
import os
from kubernetes.client.models import V1EnvVar


def add_wandb_env_variables(op):
    env = {
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        "WANDB_BASE_URL": os.getenv("WANDB_BASE_URL"),
    }

    for name, value in env.items():
        op = op.add_env_variable(V1EnvVar(name, value))
    return op


@dsl.pipeline(name="example-pipeline")
def example_pipeline(param1: str, param2: int):
    conf = dsl.get_pipeline_conf()
    conf.add_op_transformer(add_wandb_env_variables)
```

## プログラムでデータにアクセス

### Kubeflow Pipelines UI経由

W&Bでログされた Kubeflow Pipelines UIの任意の Run をクリックします。

* 入出力に関する詳細は `Input/Output` および `ML Metadata` タブで確認できます。
* `Visualizations` タブから W&B ウェブアプリを表示します。

{{< img src="/images/integrations/kubeflow_app_pipelines_ui.png" alt="Get a view of W&B in the Kubeflow UI" >}}

### ウェブアプリ UI経由

ウェブアプリ UI には Kubeflow Pipelines の `Visualizations` タブと同じ内容がありますが、より多くのスペースがあります。[こちらでウェブアプリ UI について詳しく学んでください]({{< relref path="/guides/models/app" lang="ja" >}})。

{{< img src="/images/integrations/kubeflow_pipelines.png" alt="View details about a particular run (and link back to the Kubeflow UI)" >}}

{{< img src="/images/integrations/kubeflow_via_app.png" alt="See the full DAG of inputs and outputs at each stage of your pipeline" >}}

### 公共 API 経由 (プログラムによるアクセス)

* プログラムによるアクセスには、[我々の公共 APIを参照してください]({{< relref path="/ref/python/public-api" lang="ja" >}})。

### Kubeflow Pipelines と W&B の概念マッピング

こちらは Kubeflow Pipelines の概念を W&B にマッピングしたものです。

| Kubeflow Pipelines | W&B | W&B の場所 |
| ------------------ | --- | ---------- |
| Input Scalar | [`config`]({{< relref path="/guides/models/track/config" lang="ja" >}}) | [Overview タブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) |
| Output Scalar | [`summary`]({{< relref path="/guides/models/track/log" lang="ja" >}}) | [Overview タブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) |
| Input Artifact | Input Artifact | [Artifacts タブ]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}}) |
| Output Artifact | Output Artifact | [Artifacts タブ]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}}) |

## 詳細なログ

詳細なログを取得するために、`wandb.log`および`wandb.log_artifact`コールをコンポーネントに自由に挿入できます。

### 明示的な `wandb.log_artifacts` コールを使用

以下の例では、モデルをトレーニングしています。`@wandb_log`デコレータは関連する入出力を自動的にトラックします。トレーニングプロセスのログを取るためには、以下のように明示的にログを追加することができます。

```python
@wandb_log
def train_model(
    train_dataloader_path: components.InputPath("dataloader"),
    test_dataloader_path: components.InputPath("dataloader"),
    model_path: components.OutputPath("pytorch_model"),
):
    ...
    for epoch in epochs:
        for batch_idx, (data, target) in enumerate(train_dataloader):
            ...
            if batch_idx % log_interval == 0:
                wandb.log(
                    {"epoch": epoch, "step": batch_idx * len(data), "loss": loss.item()}
                )
        ...
        wandb.log_artifact(model_artifact)
```

### 暗黙的な wandb インテグレーションを使用

[フレームワークインテグレーション]({{< relref path="/guides/integrations/" lang="ja" >}})を使用している場合は、コールバックを直接渡すこともできます。

```python
@wandb_log
def train_model(
    train_dataloader_path: components.InputPath("dataloader"),
    test_dataloader_path: components.InputPath("dataloader"),
    model_path: components.OutputPath("pytorch_model"),
):
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning import Trainer

    trainer = Trainer(logger=WandbLogger())
    ...  # トレーニングを行う
```