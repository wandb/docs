---
title: Cohere fine-tuning
description: W&B를 사용하여 Cohere 모델을 파인튜닝하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-cohere-fine-tuning
    parent: integrations
weight: 40
---

Weights & Biases를 사용하면 Cohere 모델의 미세 조정 메트릭 및 설정을 기록하여 모델의 성능을 분석하고 이해하며 동료와 결과를 공유할 수 있습니다.

이 [Cohere 가이드](https://docs.cohere.com/page/convfinqa-finetuning-wandb)에는 미세 조정 run을 시작하는 방법에 대한 전체 예제가 있으며, [Cohere API 문서](https://docs.cohere.com/reference/createfinetunedmodel#request.body.settings.wandb)는 여기에서 찾을 수 있습니다.

## Cohere 미세 조정 결과 기록

Cohere 미세 조정 로깅을 W&B workspace에 추가하려면 다음을 수행하십시오.

1. W&B API 키, W&B `entity` 및 `project` 이름으로 `WandbConfig`를 생성합니다. W&B API 키는 https://wandb.ai/authorize 에서 찾을 수 있습니다.

2. 모델 이름, 데이터셋 및 하이퍼파라미터와 함께 이 설정을 `FinetunedModel` 오브젝트에 전달하여 미세 조정 run을 시작합니다.

    ```python
    from cohere.finetuning import WandbConfig, FinetunedModel

    # W&B 세부 정보로 config를 생성합니다.
    wandb_ft_config = WandbConfig(
        api_key="<wandb_api_key>",
        entity="my-entity", # 제공된 API 키와 연결된 유효한 entity여야 합니다.
        project="cohere-ft",
    )

    ...  # 데이터셋 및 하이퍼파라미터를 설정합니다.

    # cohere에서 미세 조정 run을 시작합니다.
    cmd_r_finetune = co.finetuning.create_finetuned_model(
      request=FinetunedModel(
        name="command-r-ft",
        settings=Settings(
          base_model=...
          dataset_id=...
          hyperparameters=...
          wandb=wandb_ft_config  # 여기에 W&B config를 전달합니다.
        ),
      ),
    )
    ```

3. 모델의 미세 조정 트레이닝 및 유효성 검사 메트릭과 하이퍼파라미터를 생성한 W&B project에서 확인합니다.

    {{< img src="/images/integrations/cohere_ft.png" alt="" >}}

## Runs 구성

W&B runs는 자동으로 구성되며 job 유형, base model, 학습 속도 및 기타 하이퍼파라미터와 같은 모든 configuration 파라미터를 기준으로 필터링/정렬할 수 있습니다.

또한 runs의 이름을 바꾸거나, 노트를 추가하거나, 태그를 생성하여 그룹화할 수 있습니다.

## 참고 자료

* **[Cohere Fine-tuning Example](https://github.com/cohere-ai/notebooks/blob/kkt_ft_cookbooks/notebooks/finetuning/convfinqa_finetuning_wandb.ipynb)**
