---
title: Kubeflow パイプライン (kfp)
description: W&B を Kubeflow パイプラインと統合する方法
menu:
  default:
    identifier: ja-guides-integrations-kubeflow-pipelines-kfp
    parent: integrations
weight: 170
---

[Kubeflow Pipelines (kfp)](https://www.kubeflow.org/docs/components/pipelines/overview/) は、Docker コンテナをベースにした、移植性が高くスケーラブルな機械学習（ML）ワークフローを構築・デプロイするためのプラットフォームです。

このインテグレーションにより、ユーザーは kfp の Python 関数型コンポーネントにデコレーターを適用し、パラメータや Artifacts を W&B へ自動でログできるようになります。

この機能は `wandb==0.12.11` から有効で、`kfp<2.0.0` が必要です。

## サインアップと API キーの作成

APIキー は、あなたのマシンを W&B に認証するためのものです。ユーザープロフィールから APIキー を生成できます。

{{% alert %}}
より簡単な方法として、[W&B認証ページ](https://wandb.ai/authorize) で直接 APIキー を生成できます。表示された APIキー をコピーし、パスワードマネージャ等の安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のユーザーアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックし、表示された APIキー をコピーします。APIキー を隠すにはページを再読み込みしてください。

## `wandb` ライブラリのインストールとログイン

ローカル環境で `wandb` ライブラリをインストールし、ログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) に自身の APIキー をセットします。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールし、ログインします。

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

## コンポーネントへのデコレーター追加

`@wandb_log` デコレーターを追加し、通常通りコンポーネントを作成してください。これにより、パイプラインを実行するたびに入出力パラメータや Artifacts が自動で W&B にログされます。

```python
from kfp import components
from wandb.integration.kfp import wandb_log


@wandb_log
def add(a: float, b: float) -> float:
    return a + b


add = components.create_component_from_func(add)
```

## 環境変数をコンテナへ渡す

[環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をコンテナへ明示的に渡す必要がある場合があります。双方向リンクを実現する場合は、`WANDB_KUBEFLOW_URL` 環境変数も、Kubeflow Pipelines インスタンスのベースURL（例: `https://kubeflow.mysite.com`）として設定してください。

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

## プログラムからデータにアクセスする

### Kubeflow Pipelines の UI から

Kubeflow Pipelines UI で、W&B へログされた任意の Run をクリックします。

* `Input/Output` および `ML Metadata` タブで入出力の詳細を確認できます。
* `Visualizations` タブから W&B web アプリにアクセスできます。

{{< img src="/images/integrations/kubeflow_app_pipelines_ui.png" alt="W&B in Kubeflow UI" >}}

### Web アプリ UI から

Web アプリ UI では、Kubeflow Pipelines の `Visualizations` タブと同じ内容が、より広いスペースで表示されます。[Webアプリの詳細はこちら]({{< relref path="/guides/models/app" lang="ja" >}})。

{{< img src="/images/integrations/kubeflow_pipelines.png" alt="Run details" >}}

{{< img src="/images/integrations/kubeflow_via_app.png" alt="Pipeline DAG" >}}

### パブリックAPI経由（プログラムからアクセス）

* プログラムからアクセスしたい場合は、[パブリックAPIのドキュメント]({{< relref path="/ref/python/public-api/index.md" lang="ja" >}}) をご覧ください。

### Kubeflow PipelinesとW&Bの概念マッピング

Kubeflow Pipelines の各概念と W&B との対応表です。

| Kubeflow Pipelines | W&B | W&Bでの位置 |
| ------------------ | --- | ------------- |
| Input Scalar | [`config`]({{< relref path="/guides/models/track/config" lang="ja" >}}) | [Overviewタブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) |
| Output Scalar | [`summary`]({{< relref path="/guides/models/track/log" lang="ja" >}}) | [Overviewタブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) |
| Input Artifact | Input Artifact | [Artifactsタブ]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}}) |
| Output Artifact | Output Artifact | [Artifactsタブ]({{< relref path="/guides/models/track/runs/#artifacts-tab" lang="ja" >}}) |

## きめ細かなログ

より細かくログ出力を制御したい場合は、コンポーネント内で `wandb.log` や `wandb.log_artifact` を直接呼び出せます。

### 明示的な `wandb.log_artifacts` 呼び出しの場合

以下はモデルのトレーニングの例です。`@wandb_log` デコレーターにより基本的な入出力は自動で記録されますが、トレーニング経過も記録したい場合は、以下のように明示的なログを書き加えられます。

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

[対応しているフレームワークのインテグレーション]({{< relref path="/guides/integrations/" lang="ja" >}}) を使う場合、コールバックを直接渡すこともできます。

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