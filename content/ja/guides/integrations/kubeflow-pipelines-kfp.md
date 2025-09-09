---
title: Kubeflow パイプライン (kfp)
description: W&B を Kubeflow Pipelines と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-kubeflow-pipelines-kfp
    parent: integrations
weight: 170
---

[Kubeflow Pipelines (kfp) ](https://www.kubeflow.org/docs/components/pipelines/overview/)は、dockerコンテナに基づく移植性・スケーラビリティの高い機械学習 (ML) ワークフローを構築・デプロイするためのプラットフォームです。

このインテグレーションにより、kfp の Python 関数型コンポーネントにデコレータを適用して、パラメータと Artifacts を W&B に自動でログできます。

この機能は `wandb==0.12.11` で有効になり、`kfp<2.0.0` が必要です。

## サインアップして APIキー を作成

APIキー は、あなたのマシンを W&B に認証するためのものです。APIキー はユーザー プロフィールから生成できます。

{{% alert %}}
もっとシンプルに行うには、[W&B authorization page](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザー プロファイル アイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された APIキー をコピーします。APIキー を非表示にするには、ページを再読み込みします。

## `wandb` ライブラリをインストールしてログイン

ローカルに `wandb` ライブラリをインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) に自分の APIキー を設定します。

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

{{% tab header="Python ノートブック" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}


## コンポーネントにデコレータを付与

`@wandb_log` デコレータを追加して、いつも通りにコンポーネントを作成します。これにより、パイプラインを実行するたびに、入力/出力のパラメータと Artifacts が W&B に自動でログされます。

```python
from kfp import components
from wandb.integration.kfp import wandb_log


@wandb_log
def add(a: float, b: float) -> float:
    return a + b


add = components.create_component_from_func(add)
```

## 環境変数をコンテナに渡す

コンテナに[環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})を明示的に渡す必要がある場合があります。双方向のリンクのために、`WANDB_KUBEFLOW_URL` 環境変数に Kubeflow Pipelines インスタンスのベース URL を設定してください。例: `https://kubeflow.mysite.com`。

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

## データへプログラムからアクセス

### Kubeflow Pipelines UI から

W&B でログされた任意の Run を Kubeflow Pipelines UI でクリックします。

* 入力と出力の詳細は、`Input/Output` と `ML Metadata` タブで確認できます。
* `Visualizations` タブから W&B の Web アプリを表示できます。

{{< img src="/images/integrations/kubeflow_app_pipelines_ui.png" alt="Kubeflow UI での W&B" >}}

### Web アプリの UI から

Web アプリの UI には、Kubeflow Pipelines の `Visualizations` タブと同じ内容が、より広いスペースで表示されます。詳細は[こちらの Web アプリのガイド]({{< relref path="/guides/models/app" lang="ja" >}})をご覧ください。

{{< img src="/images/integrations/kubeflow_pipelines.png" alt="Run の詳細" >}}

{{< img src="/images/integrations/kubeflow_via_app.png" alt="パイプラインの DAG" >}}

### Public API 経由（プログラムによるアクセス）

* プログラムによるアクセスについては、[Public API を参照]({{< relref path="/ref/python/public-api/index.md" lang="ja" >}})してください。

### Kubeflow Pipelines と W&B の概念対応

Kubeflow Pipelines の概念を W&B に対応付けると次のとおりです。

| Kubeflow Pipelines | W&B | W&B 上の場所 |
| ------------------ | --- | --------------- |
| 入力スカラー | [`config`]({{< relref path="/guides/models/track/config" lang="ja" >}}) | [Overview タブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) |
| 出力スカラー | [`summary`]({{< relref path="/guides/models/track/log" lang="ja" >}}) | [Overview タブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) |
| Input Artifact | Input Artifact | [Artifacts タブ]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}}) |
| Output Artifact | Output Artifact | [Artifacts タブ]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}}) |

## より細かなログ記録

ログをより細かく制御したい場合は、コンポーネント内で適宜 `wandb.log` や `wandb.log_artifact` を呼び出してください。

### `wandb.log_artifacts` を明示的に呼び出す場合

以下の例では、モデルをトレーニングしています。`@wandb_log` デコレータは、該当する入力と出力を自動で追跡します。トレーニングの進行をログしたい場合は、次のように明示的なログを追加できます。

```python
@wandb_log
def train_model(
    train_dataloader_path: components.InputPath("dataloader"),
    test_dataloader_path: components.InputPath("dataloader"),
    model_path: components.OutputPath("pytorch_model"),
):
    with wandb.init() as run:
        ...
        for epoch in epochs:
            for batch_idx, (data, target) in enumerate(train_dataloader):
                ...
                if batch_idx % log_interval == 0:
                    run.log(
                        {"epoch": epoch, "step": batch_idx * len(data), "loss": loss.item()}
                    )
            ...
            run.log_artifact(model_artifact)
```

### 暗黙的な wandb インテグレーションを使う場合

[サポートしているフレームワークのインテグレーション]({{< relref path="/guides/integrations/" lang="ja" >}})を使っている場合は、コールバックを直接渡すこともできます。

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
    ...  # トレーニングを実行
```