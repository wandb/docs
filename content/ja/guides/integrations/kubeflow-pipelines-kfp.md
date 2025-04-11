---
title: Kubeflow パイプライン (kfp)
description: W&B を Kubeflow パイプラインと統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-kubeflow-pipelines-kfp
    parent: integrations
weight: 170
---

[https://www.kubeflow.org/docs/components/pipelines/overview/](https://www.kubeflow.org/docs/components/pipelines/overview/) は、Dockerコンテナに基づいて、移植性がありスケーラブルな機械学習（ML）ワークフローを構築およびデプロイするためのプラットフォームです。

このインテグレーションにより、ユーザーは kfp のPython機能コンポーネントにデコレーターを適用して、パラメータとArtifactsを自動的にW&Bにログすることができます。

この機能は `wandb==0.12.11` で有効になり、`kfp<2.0.0` が必要です。

## 登録してAPIキーを作成する

APIキーは、あなたのマシンをW&Bに認証します。APIキーは、ユーザープロファイルから生成できます。

{{% alert %}}
より効率的な方法として、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接行ってAPIキーを生成することができます。表示されたAPIキーをコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅のユーザープロファイルアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示されたAPIキーをコピーします。APIキーを隠すには、ページをリロードしてください。

## `wandb` ライブラリをインストールしてログイン

ローカルに `wandb` ライブラリをインストールしてログインするには：

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をAPIキーに設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールしてログインします。

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

## コンポーネントをデコレートする

`@wandb_log` デコレーターを追加し、通常通りコンポーネントを作成します。これにより、パイプラインを実行するたびに入力/出力パラメータとArtifactsがW&Bに自動的にログされます。

```python
from kfp import components
from wandb.integration.kfp import wandb_log


@wandb_log
def add(a: float, b: float) -> float:
    return a + b


add = components.create_component_from_func(add)
```

## 環境変数をコンテナに渡す

[環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})をコンテナに明示的に渡す必要があるかもしれません。双方向リンクのためには、`WANDB_KUBEFLOW_URL` 環境変数をKubeflow Pipelinesインスタンスの基本URLに設定する必要があります。例えば、`https://kubeflow.mysite.com`です。

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

## データへのプログラムによるアクセス

### Kubeflow Pipelines UI から

W&Bでログされた任意の Run を Kubeflow Pipelines UI でクリックします。

* `Input/Output` と `ML Metadata` タブで入力と出力の詳細を見つけます。
* `Visualizations` タブからW&Bウェブアプリを表示します。

{{< img src="/images/integrations/kubeflow_app_pipelines_ui.png" alt="Kubeflow UI でのW&Bのビューを取得" >}}

### ウェブアプリ UI から

ウェブアプリ UI は Kubeflow Pipelines の `Visualizations` タブと同じコンテンツを持っていますが、より多くのスペースがあります。[ここでウェブアプリ UI についてもっと学びましょう]({{< relref path="/guides/models/app" lang="ja" >}})。

{{< img src="/images/integrations/kubeflow_pipelines.png" alt="特定の run の詳細を確認し、Kubeflow UI に戻るリンク" >}}

{{< img src="/images/integrations/kubeflow_via_app.png" alt="パイプラインの各ステージでの入力と出力の完全なDAGを見る" >}}

### 公開APIを通じて（プログラムによるアクセス）

* プログラムによるアクセスのために、[私たちの公開APIをご覧ください]({{< relref path="/ref/python/public-api" lang="ja" >}})。

### Kubeflow Pipelines と W&B の概念マッピング

ここに、Kubeflow Pipelines の概念を W&B にマッピングしたものがあります。

| Kubeflow Pipelines | W&B | W&B 内の場所 |
| ------------------ | --- | --------------- |
| Input Scalar | [`config`]({{< relref path="/guides/models/track/config" lang="ja" >}}) | [Overview tab]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) |
| Output Scalar | [`summary`]({{< relref path="/guides/models/track/log" lang="ja" >}}) | [Overview tab]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) |
| Input Artifact | Input Artifact | [Artifacts tab]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}}) |
| Output Artifact | Output Artifact | [Artifacts tab]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}}) |

## 細かいログ

ログのコントロールを細かくしたい場合は、コンポーネントに `wandb.log` と `wandb.log_artifact` の呼び出しを追加できます。

### 明示的な `wandb.log_artifacts` 呼び出しと共に

以下の例では、モデルをトレーニングしています。`@wandb_log` デコレーターは関連する入力と出力を自動的に追跡します。トレーニングプロセスをログに追加したい場合は、以下のようにそのログを明示的に追加できます。

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

もしサポートする [フレームワークインテグレーションを使用]({{< relref path="/guides/integrations/" lang="ja" >}}) している場合は、コールバックを直接渡すこともできます。

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