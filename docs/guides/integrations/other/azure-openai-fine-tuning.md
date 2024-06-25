---
description: W&B を使用して Azure OpenAI モデルをファインチューンする方法
slug: /guides/integrations/azure-openai-fine-tuning
displayed_sidebar: default
---


# Azure OpenAI ファインチューニング

## Introduction
Microsoft Azure上でGPT-3.5またはGPT-4モデルをW&Bを使ってファインチューニングすることで、モデルのパフォーマンスを詳細に追跡し、分析できます。このガイドでは、[OpenAI Fine-Tuning guide](/guides/integrations/openai) の概念を拡張し、Azure OpenAI向けの特定の手順と機能を紹介します。

![](/images/integrations/open_ai_auto_scan.png)

:::info
Weights and Biasesのファインチューンインテグレーションは`openai >= 1.0`で動作します。最新バージョンの`openai`をインストールするには、`pip install -U openai`を実行してください。
:::

## Prerequisites
- [公式のAzureドキュメント](https://learn.microsoft.com/en-us/azure/ai-services/openai/tutorials/fine-tune)に従って設定されたAzure OpenAIサービス。
- 最新バージョンの`openai`、`wandb`、および必要な他のライブラリがインストールされていること。

## W&BでAzure OpenAIファインチューン結果を2行で同期

```python
from openai import AzureOpenAI

# Azure OpenAIに接続
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
)

# トレーニングおよびバリデーションのデータセットをJSONL形式で作成および検証し、
# クライアント経由でアップロードし、
# ファインチューニングジョブを開始します。

from wandb.integration.openai.fine_tuning import WandbLogger

# ファインチューニング結果をW&Bと同期！
WandbLogger.sync(
    fine_tune_job_id=job_id, openai_client=client, project="your_project_name"
)
```

### インタラクティブな例を見る

* [Demo Colab](http://wandb.me/azure-openai-colab)

## W&Bでの可視化とバージョン管理
- Tablesとしてトレーニングとバリデーションデータをバージョン管理し、可視化するためにW&Bを利用。
- データセットとモデルのメタデータはW&B Artifactsとしてバージョン管理され、効率的な追跡とバージョンコントロールが可能。

![](/images/integrations/openai_data_artifacts.png)

![](/images/integrations/openai_data_visualization.png)

## ファインチューンしたモデルの取得
- ファインチューンしたモデルIDはAzure OpenAIから取得可能で、モデルのメタデータの一部としてW&Bにログされます。

![](/images/integrations/openai_model_metadata.png)

## Additional resources
- [OpenAI Fine-tuning Documentation](https://platform.openai.com/docs/guides/fine-tuning/)
- [Azure OpenAI Fine-tuning Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/fine-tuning?tabs=turbo%2Cpython&pivots=programming-language-python)
- [Demo Colab](http://wandb.me/azure-openai-colab)