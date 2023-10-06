---
slug: /guides/integrations/kubeflow-pipelines-kfp
description: How to integrate W&B with Kubeflow Pipelines.
displayed_sidebar: ja
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Kubeflow Pipelines (kfp)

## 概要

[Kubeflow Pipelines (kfp)](https://www.kubeflow.org/docs/components/pipelines/introduction/)は、Dockerコンテナをベースとしたポータブルでスケーラブルな機械学習（ML）ワークフローを構築・デプロイするプラットフォームです。

この統合により、ユーザーはkfpのPython関数コンポーネントにデコレータを適用して、パラメータとアーティファクトをW&Bに自動的に記録することができます。

この機能は`wandb==0.12.11`で有効になり、`kfp<2.0.0`が必要です。

## クイックスタート

### W&Bのインストールとログイン

<Tabs
  defaultValue="notebook"
  values={[
    {label: 'ノートブック', value: 'notebook'},
    {label: 'コマンドライン', value: 'cli'},
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

### コンポーネントにデコレータを追加

`@wandb_log` デコレータを追加し、コンポーネントを通常どおり作成します。これにより、パイプラインを実行するたびに、入力/出力パラメータとアーティファクトが自動的にW&Bにログされます。

```python
from kfp import components
from wandb.integration.kfp import wandb_log

@wandb_log
def add(a: float, b: float) -> float:
    return a + b
```

add = components.create_component_from_func(add)
```

### コンテナに環境変数を渡す

コンテナに[WANDB環境変数](../../track/environment-variables.md)を明示的に渡す必要があるかもしれません。双方向リンクの場合は、環境変数`WANDB_KUBEFLOW_URL`にKubeflow Pipelinesインスタンスの基本URL（例: https://kubeflow.mysite.com）を設定する必要があります。

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

## 私のデータはどこにありますか？プログラムでアクセスできますか？
### Kubeflow Pipelines UI経由

W&Bでログを記録したKubeflow Pipelines UIの任意のRunをクリックします。

* 入力と出力は、`Input/Output`タブと`ML Metadata`タブでトラッキングされます。
* `Visualizations`タブからW&Bウェブアプリを表示することもできます。

![Kubeflow UIでW&Bを表示する](/images/integrations/kubeflow_app_pipelines_ui.png)

### ウェブアプリUI経由

ウェブアプリUIは、Kubeflow Pipelinesの`Visualizations`タブと同じ内容ですが、スペースがもっと広いです！[ウェブアプリUIについての詳細はこちら](https://docs.wandb.ai/ref/app)。

![特定のRunの詳細を表示する（Kubeflow UIへのリンクもあります）](/images/integrations/kubeflow_pipelines.png)

![パイプラインの各段階での入力と出力の完全なDAGを表示する](/images/integrations/kubeflow_via_app.png)

### Public API経由（プログラムでのアクセス）

* プログラムでのアクセスのために、[Public APIをご覧ください](https://docs.wandb.ai/ref/python/public-api)。

### Kubeflow PipelinesからW&Bへの概念マッピング

以下は、Kubeflow Pipelinesの概念をW&Bにマッピングしたものです。

| Kubeflow Pipelines | W&B                                                      | W&B内の位置                                                                                      |
| ------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 入力スカラー       | ``[`config`](https://docs.wandb.ai/guides/track/config)`` | [概要タブ](https://docs.wandb.ai/ref/app/pages/run-page#overview-tab)                             |
| 出力スカラー       | ``[`summary`](https://docs.wandb.ai/guides/track/log)``   | [概要タブ](https://docs.wandb.ai/ref/app/pages/run-page#overview-tab)                             |
| 入力アーティファクト | 入力アーティファクト                                       | [アーティファクトタブ](https://docs.wandb.ai/ref/app/pages/run-page#artifacts-tab)               |
| 出力アーティファクト | 出力アーティファクト                                       | [アーティファクトタブ](https://docs.wandb.ai/ref/app/pages/run-page#artifacts-tab) |
## 細かいログ記録

ログ記録をより細かく制御したい場合は、`wandb.log`および`wandb.log_artifact`をコンポーネントに追加できます。

### 明示的なwandbログ記録呼び出しを使用して

以下の例では、モデルをトレーニングしています。`@wandb_log`デコレータは関連する入力と出力を自動的にトラッキングします。トレーニングプロセスをログに記録したい場合は、以下のようにログを明示的に追加できます。

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
### 暗黙的なwandbの統合



[対応しているフレームワークの統合](https://docs.wandb.ai/guides/integrations)を使用している場合は、コールバックを直接渡すこともできます。



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

    ...  # トレーニングを行う

```