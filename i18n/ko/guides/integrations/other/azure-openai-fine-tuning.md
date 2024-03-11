---
description: How to Fine-Tune Azure OpenAI models using W&B.
slug: /guides/integrations/azure-openai-fine-tuning
displayed_sidebar: default
---

# Azure OpenAI 파인튜닝

## 도입
Microsoft Azure에서 W&B를 사용하여 GPT-3.5 또는 GPT-4 모델을 파인튜닝하는 것은 모델 성능의 자세한 추적 및 분석을 가능하게 합니다. 이 가이드는 [OpenAI 파인튜닝 가이드](/guides/integrations/openai)에서 다룬 개념을 Azure OpenAI에 특화된 단계와 기능으로 확장합니다.

![](/images/integrations/open_ai_auto_scan.png)

:::안내
Weights and Biases 파인튜닝 인테그레이션은 `openai >= 1.0`과 함께 작동합니다. 최신 버전의 `openai`를 설치하려면 `pip install -U openai`를 실행하세요.
:::

## 전제 조건
- [공식 Azure 문서](https://learn.microsoft.com/en-us/azure/ai-services/openai/tutorials/fine-tune)에 따라 설정된 Azure OpenAI 서비스.
- `openai`, `wandb`, 그리고 다른 필요한 라이브러리의 최신 버전이 설치되어 있어야 합니다.

## W&B에서 Azure OpenAI 파인튜닝 결과를 2줄로 동기화하기

```python
from openai import AzureOpenAI

# Azure OpenAI에 연결하기
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
)

# 트레이닝 및 검증 데이터셋을 JSONL 포맷으로 생성 및 검증하고,
# 클라이언트를 통해 업로드하고,
# 파인튜닝 작업을 시작합니다.

from wandb.integration.openai.fine_tuning import WandbLogger

# W&B와 파인튜닝 결과를 동기화하세요!
WandbLogger.sync(
    fine_tune_job_id=job_id, openai_client=client, project="your_project_name"
)
```

### 인터랙티브 예시 확인하기

* [데모 Colab](http://wandb.me/azure-openai-colab)

## W&B에서의 시각화 및 버전 관리
- 테이블로 트레이닝 및 검증 데이터를 버전 관리하고 시각화하기 위해 W&B를 활용하세요.
- 데이터셋과 모델 메타데이터는 W&B 아티팩트로 버전이 관리되어 효율적인 추적 및 버전 제어가 가능합니다.

![](/images/integrations/openai_data_artifacts.png)

![](/images/integrations/openai_data_visualization.png)

## 파인튜닝된 모델 검색
- 파인튜닝된 모델 ID는 Azure OpenAI에서 검색할 수 있으며 W&B에서 모델 메타데이터의 일부로 로그됩니다.

![](/images/integrations/openai_model_metadata.png)

## 추가 자료
- [OpenAI 파인튜닝 문서](https://platform.openai.com/docs/guides/fine-tuning/)
- [Azure OpenAI 파인튜닝 문서](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/fine-tuning?tabs=turbo%2Cpython&pivots=programming-language-python)
- [데모 Colab](http://wandb.me/azure-openai-colab)