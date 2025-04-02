---
title: Kubeflow Pipelines (kfp)
description: W&B と Kubeflow Pipelines を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-kubeflow-pipelines-kfp
    parent: integrations
weight: 170
---

[Kubeflow Pipelines (kfp) ](https://www.kubeflow.org/docs/components/pipelines/overview/) は、Docker コンテナに基づいて、移植可能でスケーラブルな 機械学習 (ML) ワークフローを構築およびデプロイするためのプラットフォームです。

このインテグレーションにより、ユーザーはデコレーターを kfp python 関数コンポーネントに適用して、 パラメータと Artifacts を W&B に自動的に記録できます。

この機能は `wandb==0.12.11` で有効になり、`kfp<2.0.0` が必要です。

## サインアップして API キーを作成する

API キーは、お使いのマシンを W&B に対して認証します。API キーは、ユーザープロファイルから生成できます。

{{% alert %}}
より効率的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして API キーを生成できます。表示された API キーをコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅にあるユーザープロファイルアイコンをクリックします。
2. [**User Settings**] を選択し、[**API Keys**] セクションまでスクロールします。
3. [**Reveal**] をクリックします。表示された API キーをコピーします。API キーを非表示にするには、ページをリロードします。

## `wandb` ライブラリをインストールしてログインする

`wandb` ライブラリをローカルにインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. API キーに `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` ライブラリをインストールしてログインします。

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

`@wandb_log` デコレーターを追加し、通常どおりにコンポーネントを作成します。これにより、パイプラインを実行するたびに、入力/出力 パラメータ と Artifacts が自動的に W&B に記録されます。

```python
from kfp import components
from wandb.integration.kfp import wandb_log


@wandb_log
def add(a: float, b: float) -> float:
    return a + b


add = components.create_component_from_func(add)
```

## 環境変数をコンテナに渡す

[環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をコンテナに明示的に渡す必要がある場合があります。双方向のリンクを行うには、環境変数 `WANDB_KUBEFLOW_URL` を Kubeflow Pipelines インスタンスのベース URL に設定する必要があります。たとえば、`https://kubeflow.mysite.com` です。

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

## プログラムでデータにアクセスする

### Kubeflow Pipelines UI 経由

W&B でログに記録された Kubeflow Pipelines UI の Run をクリックします。

* [ `Input/Output` ] タブと [ `ML Metadata` ] タブで、入力と出力に関する詳細を確認します。
* [ `Visualizations` ] タブから W&B Web アプリを表示します。

{{< img src="/images/integrations/kubeflow_app_pipelines_ui.png" alt="Kubeflow UI で W&B の表示を取得する" >}}

### Web アプリ UI 経由

Web アプリ UI には、Kubeflow Pipelines の [ `Visualizations` ] タブと同じコンテンツがありますが、より広いスペースがあります。[Web アプリ UI の詳細はこちら]({{< relref path="/guides/models/app" lang="ja" >}})をご覧ください。

{{< img src="/images/integrations/kubeflow_pipelines.png" alt="特定の Run に関する詳細を表示する (そして Kubeflow UI にリンクバックする)" >}}

{{< img src="/images/integrations/kubeflow_via_app.png" alt="パイプラインの各段階での入力と出力の完全な DAG を表示する" >}}

### パブリック API 経由 (プログラムによるアクセス)

* プログラムによるアクセスについては、[パブリック API を参照してください]({{< relref path="/ref/python/public-api" lang="ja" >}})。

### Kubeflow Pipelines から W&B へのコンセプトマッピング

Kubeflow Pipelines のコンセプトから W&B へのマッピングを次に示します。

| Kubeflow Pipelines | W&B | W&B の場所 |
| ------------------ | --- | --------------- |
| 入力スカラー | [`config`]({{< relref path="/guides/models/track/config" lang="ja" >}}) | [Overviewタブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) |
| 出力スカラー | [`summary`]({{< relref path="/guides/models/track/log" lang="ja" >}}) | [Overviewタブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) |
| 入力 Artifact | 入力 Artifact | [Artifacts タブ]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}}) |
| 出力 Artifact | 出力 Artifact | [Artifacts タブ]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}}) |

## きめ細かいロギング

ロギングをより細かく制御したい場合は、コンポーネントに `wandb.log` および `wandb.log_artifact` 呼び出しを追加できます。

### 明示的な `wandb.log_artifacts` 呼び出しを使用

以下の例では、モデルをトレーニングしています。`@wandb_log` デコレーターは、関連する入力と出力を自動的に追跡します。トレーニングプロセスをログに記録する場合は、次のように明示的にロギングを追加できます。

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

[サポートするフレームワーク インテグレーション]({{< relref path="/guides/integrations/" lang="ja" >}}) を使用している場合は、コールバックを直接渡すこともできます。

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
    ...  # do training
```
