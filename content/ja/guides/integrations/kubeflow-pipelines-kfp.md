---
title: Kubeflow パイプライン (kfp)
description: Kubeflow パイプラインと W&B を連携する方法
menu:
  default:
    identifier: kubeflow-pipelines-kfp
    parent: integrations
weight: 170
---

[Kubeflow Pipelines (kfp)](https://www.kubeflow.org/docs/components/pipelines/overview/) は、Docker コンテナをベースにした持ち運び可能でスケーラブルな機械学習 (ML) ワークフローを構築・デプロイするためのプラットフォームです。

このインテグレーションにより、ユーザーは kfp の Python 関数型コンポーネントにデコレータを適用し、パラメータやArtifactsの自動ロギングを W&B に行うことができます。

この機能は `wandb==0.12.11` から利用可能で、`kfp<2.0.0` が必要です。

## サインアップと APIキー の作成

APIキーは、あなたのマシンを W&B に認証するものです。APIキーはユーザープロフィールから発行できます。

{{% alert %}}
もっと簡単な方法として、[W&B認証ページ](https://wandb.ai/authorize)に直接アクセスして APIキー を発行できます。表示された APIキー をコピーし、パスワードマネージャーなど安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックし、表示された APIキー をコピーします。APIキー を隠したい場合は、ページをリロードしてください。

## `wandb` ライブラリのインストールとログイン

ローカルで `wandb` ライブラリをインストールしログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref "/guides/models/track/environment-variables.md" >}})をAPIキーで設定します。

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

`@wandb_log` デコレータを追加し、通常通りにコンポーネントを作成します。こうすることで、パイプラインを実行する度に入力／出力のパラメータとArtifactsが自動で W&B に記録されます。

```python
from kfp import components
from wandb.integration.kfp import wandb_log

@wandb_log
def add(a: float, b: float) -> float:
    return a + b

add = components.create_component_from_func(add)
```

## 環境変数をコンテナに渡す

[環境変数]({{< relref "/guides/models/track/environment-variables.md" >}})をコンテナに明示的に渡す必要がある場合があります。双方向のリンクには、`WANDB_KUBEFLOW_URL` 環境変数も Kubeflow Pipelines インスタンスのベースURLに設定してください。例: `https://kubeflow.mysite.com`。

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

## データへプログラムでアクセスする

### Kubeflow Pipelines UI 経由

W&B でロギングされた Kubeflow Pipelines UI の任意の Run をクリックします。

* `Input/Output` タブや `ML Metadata` タブで入力値・出力値の詳細が確認できます。
* `Visualizations` タブからW&Bウェブアプリを開けます。

{{< img src="/images/integrations/kubeflow_app_pipelines_ui.png" alt="W&B in Kubeflow UI" >}}

### ウェブアプリ UI 経由

ウェブアプリのUIは、Kubeflow Pipelines の `Visualizations` タブと同じ内容ですが、より広いスペースで確認できます。[ウェブアプリ UI の詳細はこちら]({{< relref "/guides/models/app" >}})。

{{< img src="/images/integrations/kubeflow_pipelines.png" alt="Run details" >}}

{{< img src="/images/integrations/kubeflow_via_app.png" alt="Pipeline DAG" >}}

### パブリックAPI経由（プログラムによるアクセス）

* プログラムによるアクセスの詳細は [Public API]({{< relref "/ref/python/public-api/index.md" >}}) をご覧ください。

### Kubeflow Pipelines から W&B へのコンセプトマッピング

Kubeflow Pipelines と W&B の対応関係は次の通りです。

| Kubeflow Pipelines | W&B | W&B内での場所 |
| ------------------ | --- | --------------- |
| Input Scalar | [`config`]({{< relref "/guides/models/track/config" >}}) | [Overviewタブ]({{< relref "/guides/models/track/runs/#overview-tab" >}}) |
| Output Scalar | [`summary`]({{< relref "/guides/models/track/log" >}}) | [Overviewタブ]({{< relref "/guides/models/track/runs/#overview-tab" >}}) |
| Input Artifact | Input Artifact | [Artifactsタブ]({{< relref "/guides/models/track/runs/#artifacts-tab" >}}) |
| Output Artifact | Output Artifact | [Artifactsタブ]({{< relref "/guides/models/track/runs/#artifacts-tab" >}}) |

## きめ細かなロギング

さらに詳細なログを取りたい場合、コンポーネント内で `wandb.log` や `wandb.log_artifact` を直接呼び出して記録することもできます。

### 明示的な `wandb.log_artifact` 呼び出し

以下の例では、モデルのトレーニングを行っています。`@wandb_log` デコレータによって主要な入力・出力は自動で追跡されますが、トレーニング過程のログを追加したい場合は下記のように記述できます。

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

### 暗黙的な wandb インテグレーションの利用

[サポートされているフレームワークインテグレーション]({{< relref "/guides/integrations/" >}})を利用する場合、コールバックを直接渡すこともできます。

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
    ...  # ここでトレーニングを実行
```
