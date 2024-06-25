---
description: W&BをKubeflowパイプラインと統合する方法。
slug: /guides/integrations/kubeflow-pipelines-kfp
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Kubeflow Pipelines (kfp)

## 概要

[Kubeflow Pipelines (kfp) ](https://www.kubeflow.org/docs/components/pipelines/introduction/)は、Dockerコンテナに基づいたポータブルでスケーラブルな機械学習（ML）ワークフローを構築および展開するためのプラットフォームです。

このインテグレーションにより、kfpのPythonファンクショナルコンポーネントにデコレータを適用して、パラメータやArtifactsを自動的にW&Bにログすることができます。

この機能は `wandb==0.12.11` で有効になり、`kfp<2.0.0`が必要です。

## クイックスタート

### W&Bのインストールとログイン

<Tabs
  defaultValue="notebook"
  values={[
    {label: 'Notebook', value: 'notebook'},
    {label: 'Command Line', value: 'cli'},
  ]}>
  <TabItem value="notebook">

```python
!pip install kfp wandb

import wandb
wandb.login()
```

  </TabItem>
  <TabItem value="cli">

```
pip install kfp wandb
wandb login
```

  </TabItem>
</Tabs>

### コンポーネントにデコレート

`@wandb_log`デコレータを追加し、通常通りにコンポーネントを作成します。これにより、パイプラインを実行するたびに入出力パラメータとArtifactsが自動的にW&Bにログされます。

```python
from kfp import components
from wandb.integration.kfp import wandb_log

@wandb_log
def add(a: float, b: float) -> float:
    return a + b

add = components.create_component_from_func(add)
```

### 環境変数をコンテナに渡す

必要に応じて[WANDB環境変数](../../track/environment-variables.md)をコンテナに明示的に渡すことができます。双方向リンクのために、環境変数 `WANDB_KUBEFLOW_URL` をKubeflow Pipelinesインスタンスの基本URL（例：https://kubeflow.mysite.com）に設定する必要があります。

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
def example_pipeline(...):
    conf = dsl.get_pipeline_conf()
    conf.add_op_transformer(add_wandb_env_variables)
    ...
```

## データはどこにありますか？プログラムでアクセスできますか？

### Kubeflow Pipelines UI経由

W&BでログされたKubeflow Pipelines UIの任意のRunをクリックしてください。

* 入力と出力は`Input/Output`および`ML Metadata`タブで追跡されます。
* `Visualizations`タブからW&B webアプリを表示することもできます。

![Get a view of W&B in the Kubeflow UI](/images/integrations/kubeflow_app_pipelines_ui.png)

### WebアプリUI経由

WebアプリUIはKubeflow Pipelinesの`Visualizations`タブと同じ内容を持っていますが、より広いスペースを提供します！ 詳細は[こちら](https://docs.wandb.ai/ref/app)をご覧ください。

![View details about a particular run (and link back to the Kubeflow UI)](/images/integrations/kubeflow_pipelines.png)

![See the full DAG of inputs and outputs at each stage of your pipeline](/images/integrations/kubeflow_via_app.png)

### 公開API経由 (プログラムからのアクセス)

* プログラムからのアクセスについては、[公開API](https://docs.wandb.ai/ref/python/public-api)をご覧ください。

### Kubeflow PipelinesからW&Bへの概念マッピング

こちらにKubeflow Pipelinesの概念をW&Bにマッピングしたものを示します。

| Kubeflow Pipelines | W&B                                                      | W&Bの場所                                                                                     |
| ------------------ | --------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Input Scalar       | ``[`config`](https://docs.wandb.ai/guides/track/config)`` | [Overview tab](https://docs.wandb.ai/ref/app/pages/run-page#overview-tab)                    |
| Output Scalar      | ``[`summary`](https://docs.wandb.ai/guides/track/log)``   | [Overview tab](https://docs.wandb.ai/ref/app/pages/run-page#overview-tab)                    |
| Input Artifact     | Input Artifact                                            | [Artifacts tab](https://docs.wandb.ai/ref/app/pages/run-page#artifacts-tab)                  |
| Output Artifact    | Output Artifact                                           | [Artifacts tab](https://docs.wandb.ai/ref/app/pages/run-page#artifacts-tab)                  |

## 細かいログの記録

細かい制御が必要な場合は、`wandb.log`および`wandb.log_artifact`を使用して、コンポーネント内でログを記録することができます。

### 明示的なwandbログ呼び出し

以下の例では、モデルをトレーニングしています。`@wandb_log`デコレータは、関連する入力と出力を自動的に追跡します。トレーニングプロセスをログしたい場合は、以下のようにしてログを追加できます。

```python
@wandb_log
def train_model(
    train_dataloader_path: components.InputPath("dataloader"),
    test_dataloader_path: components.InputPath("dataloader"),
    model_path: components.OutputPath("pytorch_model")
):
    ...
    for epoch in epochs:
        for batch_idx, (data, target) in enumerate(train_dataloader):
            ...
            if batch_idx % log_interval == 0:
                wandb.log({
                    "epoch": epoch,
                    "step": batch_idx * len(data),
                    "loss": loss.item()
                })
        ...
        wandb.log_artifact(model_artifact)
```

### 暗黙的なwandbインテグレーションを使用

使用中の[フレームワークインテグレーション](https://docs.wandb.ai/guides/integrations)がサポートされている場合、コールバックを直接渡すこともできます。

```python
@wandb_log
def train_model(
    train_dataloader_path: components.InputPath("dataloader"),
    test_dataloader_path: components.InputPath("dataloader"),
    model_path: components.OutputPath("pytorch_model")
):
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning import Trainer
    
    trainer = Trainer(logger=WandbLogger())
    ...  # トレーニングを実行
```
