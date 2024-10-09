---
title: Azure OpenAI Fine-Tuning
description: W&B를 사용하여 Azure OpenAI 모델을 파인튜닝하는 방법.
slug: /guides/integrations/azure-openai-fine-tuning
displayed_sidebar: default
---

## Introduction
Microsoft Azure에서 W&B를 사용하여 GPT-3.5 또는 GPT-4 모델을 파인튜닝하면 모델 성능을 세부적으로 추적하고 분석할 수 있습니다. 이 가이드는 Azure OpenAI에 대한 특정 단계와 기능을 포함하여 [OpenAI 파인튜닝 가이드](/guides/integrations/openai)의 개념을 확장합니다.

![](/images/integrations/open_ai_auto_scan.png)

:::info
Weights and Biases의 파인튜닝 인테그레이션은 `openai >= 1.0`과 함께 작동합니다. `pip install -U openai`를 통해 최신 버전의 `openai`를 설치하세요.
:::

## Prerequisites
- [공식 Azure 문서](https://learn.microsoft.com/en-us/azure/ai-services/openai/tutorials/fine-tune)에 따라 설정된 Azure OpenAI 서비스.
- 최신 버전의 `openai`, `wandb`, 및 기타 필요한 라이브러리 설치.

## W&B에서 Azure OpenAI 파인튜닝 결과를 2줄로 동기화

```python
from openai import AzureOpenAI

# Azure OpenAI에 연결
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
)

# 트레이닝과 검증 데이터셋을 JSONL 형식으로 생성 및 검증하고,
# 클라이언트를 통해 업로드하고,
# 파인튜닝 작업을 시작하세요.

from wandb.integration.openai.fine_tuning import WandbLogger

# 당신의 파인튜닝 결과를 W&B와 동기화하세요!
WandbLogger.sync(
    fine_tune_job_id=job_id, openai_client=client, project="your_project_name"
)
```

### 인터랙티브 예제를 확인하세요

* [Demo Colab](http://wandb.me/azure-openai-colab)

## Visualization and versioning in W&B
- 테이블로 트레이닝 및 검증 데이터를 버전 관리하고 시각화하려면 W&B를 이용하세요.
- 데이터셋과 모델 메타데이터는 W&B Artifacts로 버전 관리되어 효율적인 추적과 버전 컨트롤이 가능합니다.

![](/images/integrations/openai_data_artifacts.png)

![](/images/integrations/openai_data_visualization.png)

## 파인튠된 모델 검색
- 파인튠된 모델 ID는 Azure OpenAI에서 검색 가능하며, W&B의 모델 메타데이터의 일부로 로그됩니다.

![](/images/integrations/openai_model_metadata.png)

## Additional resources
- [OpenAI Fine-tuning Documentation](https://platform.openai.com/docs/guides/fine-tuning/)
- [Azure OpenAI Fine-tuning Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/fine-tuning?tabs=turbo%2Cpython&pivots=programming-language-python)
- [Demo Colab](http://wandb.me/azure-openai-colab)