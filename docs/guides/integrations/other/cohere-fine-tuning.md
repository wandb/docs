---
description: Cohere 모델을 W&B로 파인튜닝하는 방법.
slug: /guides/integrations/cohere-fine-tunining
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Cohere 파인튜닝

Weights & Biases를 사용하면 Cohere 모델의 파인튜닝 메트릭과 설정을 로그하여 모델의 성능을 분석하고 이해하며, 결과를 동료와 공유할 수 있습니다.

[Cohere 가이드](https://docs.cohere.com/page/convfinqa-finetuning-wandb)에는 파인튜닝 run을 시작하는 방법에 대한 전체 예제가 있으며, [Cohere API 문서](https://docs.cohere.com/reference/createfinetunedmodel#request.body.settings.wandb)를 여기서 찾을 수 있습니다.

## Cohere 파인튜닝 결과 로그하기

W&B 워크스페이스에 Cohere 파인튜닝 로그를 추가하려면:

1. W&B API 키, W&B `entity` 및 `project` 이름을 포함한 `WandbConfig`를 생성합니다. W&B API 키는 https://wandb.ai/authorize 에서 찾을 수 있습니다.

2. 이 설정을 `FinetunedModel` 오브젝트에 모델 이름, 데이터셋 및 하이퍼파라미터와 함께 전달하여 파인튜닝 run을 시작합니다.

```python
from cohere.finetuning import WandbConfig, FinetunedModel

# W&B 세부 정보를 사용하여 설정 생성
wandb_ft_config = WandbConfig(
    api_key="<wandb_api_key>",
    entity="my-entity", # 제공된 API 키와 연관된 유효한 entity 여야 합니다
    project="cohere-ft",
)

...  # 데이터셋 및 하이퍼파라미터 설정

# cohere에서 파인튜닝 run 시작
cmd_r_finetune = co.finetuning.create_finetuned_model(
  request=FinetunedModel(
    name="command-r-ft",
    settings=Settings(
      base_model=...
      dataset_id=...
      hyperparameters=...
      wandb=wandb_ft_config  # 여기에 W&B 설정 전달
    ),
  ),
)
```

그런 다음 생성한 W&B 프로젝트에서 모델의 파인튜닝 트레이닝과 검증 메트릭 및 하이퍼파라미터를 볼 수 있습니다.

![](/images/integrations/cohere_ft.png)

## 자주 묻는 질문

### run을 어떻게 구성할 수 있나요?

W&B run은 자동으로 구성되며, job 유형, 기본 모델, 학습률 및 기타 하이퍼파라미터와 같은 설정 파라미터를 기반으로 필터링/정렬할 수 있습니다.

추가적으로, run 이름 변경, 노트 추가 또는 태그를 만들어 그룹화할 수 있습니다.

## 리소스

* **[Cohere 파인튜닝 예제](https://github.com/cohere-ai/notebooks/blob/kkt_ft_cookbooks/notebooks/finetuning/convfinqa_finetuning_wandb.ipynb)**