---
title: Kubeflow Pipelines (kfp)
description: W&B と Kubeflow Pipelines を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-kubeflow-pipelines-kfp
    parent: integrations
weight: 170
---

[Kubeflow Pipelines (kfp) ](https://www.kubeflow.org/docs/components/pipelines/overview/) は、 Docker コンテナをベースとした、ポータブルでスケーラブルな 機械学習 (ML) の ワークフローを構築、デプロイするための プラットフォームです。

このインテグレーションにより、 ユーザーは kfp python の機能コンポーネントにデコレーターを適用して、 パラメータと Artifacts を自動的に W&B に ログ記録できます。

この機能は `wandb==0.12.11` で有効になり、 `kfp<2.0.0` が必要です。

## サインアップと APIキー の作成

APIキー は、お使いのマシンを W&B に対して認証します。 APIキー は、 ユーザー プロフィールから生成できます。

{{% alert %}}
より効率的なアプローチとして、 [https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーし、 パスワード マネージャーなどの安全な場所に保存します。
{{% /alert %}}

1. 右上隅にある ユーザー プロフィール アイコンをクリックします。
2. **ユーザー 設定** を選択し、 **APIキー** セクションまでスクロールします。
3. **表示** をクリックします。表示された APIキー をコピーします。 APIキー を非表示にするには、 ページをリロードします。

## `wandb` ライブラリ をインストールしてログインする

`wandb` ライブラリ をローカルにインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. APIキー を `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) に設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` ライブラリ をインストールしてログインします。

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

## コンポーネントのデコレート

`@wandb_log` デコレーターを追加し、 通常どおりにコンポーネントを作成します。 これにより、 パイプライン を実行するたびに、 入力/出力 パラメータ と Artifacts が自動的に W&B に ログ記録されます。

```python
from kfp import components
from wandb.integration.kfp import wandb_log


@wandb_log
def add(a: float, b: float) -> float:
    return a + b


add = components.create_component_from_func(add)
```

## 環境変数 を コンテナ に渡す

[環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を コンテナ に明示的に渡す必要がある場合があります。 双方向リンクの場合は、 Kubeflow Pipelines インスタンス のベース URL に 環境変数 `WANDB_KUBEFLOW_URL` も設定する必要があります。 たとえば、 `https://kubeflow.mysite.com` のようにします。

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

## プログラムで データ に アクセス する

### Kubeflow Pipelines UI 経由

W&B で ログ記録された Kubeflow Pipelines UI で Run をクリックします。

* `Input/Output` タブと `ML Metadata` タブで、 入力と出力に関する詳細を確認します。
* `Visualizations` タブから W&B web アプリ を表示します。

{{< img src="/images/integrations/kubeflow_app_pipelines_ui.png" alt="Get a view of W&B in the Kubeflow UI" >}}

### web アプリ UI 経由

web アプリ UI には、 Kubeflow Pipelines の `Visualizations` タブと同じコンテンツが表示されますが、 より広いスペースがあります。 [web アプリ UI の詳細はこちら]({{< relref path="/guides/models/app" lang="ja" >}}) をご覧ください。

{{< img src="/images/integrations/kubeflow_pipelines.png" alt="View details about a particular run (and link back to the Kubeflow UI)" >}}

{{< img src="/images/integrations/kubeflow_via_app.png" alt="See the full DAG of inputs and outputs at each stage of your pipeline" >}}

### Public API 経由 (プログラムによる アクセス 用)

* プログラムによる アクセス については、 [Public API をご覧ください]({{< relref path="/ref/python/public-api" lang="ja" >}})。

### Kubeflow Pipelines から W&B へのコンセプト マッピング

Kubeflow Pipelines のコンセプト から W&B へのマッピングを以下に示します。

| Kubeflow Pipelines | W&B | W&B での場所 |
| ------------------ | --- | --------------- |
| 入力 スカラー | [`config`]({{< relref path="/guides/models/track/config" lang="ja" >}}) | [Overviewタブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) |
| 出力 スカラー | [`summary`]({{< relref path="/guides/models/track/log" lang="ja" >}}) | [Overviewタブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) |
| 入力 Artifact | Input Artifact | [Artifactsタブ]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}}) |
| 出力 Artifact | Output Artifact | [Artifactsタブ]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}}) |

## きめ細かい ログ記録

ログ記録 をより細かく制御する場合は、 コンポーネントに `wandb.log` と `wandb.log_artifact` の 呼び出しを散りばめることができます。

### 明示的な `wandb.log_artifacts` 呼び出しを使用する

以下の例では、 モデル を トレーニング しています。 `@wandb_log` デコレーターは、 関連する入力と出力を自動的に追跡します。 トレーニング プロセス を ログ記録 したい場合は、 次のように明示的に ログ記録 を追加できます。

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

### 暗黙的な wandb インテグレーション を使用する

[サポートしている フレームワーク インテグレーション]({{< relref path="/guides/integrations/" lang="ja" >}}) を使用している場合は、 コールバック を直接渡すこともできます。

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
