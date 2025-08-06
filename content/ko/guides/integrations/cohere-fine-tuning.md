---
title: Cohere 파인튜닝
description: W&B를 사용하여 Cohere 모델을 파인튜닝하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-cohere-fine-tuning
    parent: integrations
weight: 40
---

W&B를 사용하면 Cohere 모델의 파인튜닝 메트릭과 설정을 로그로 남겨 모델의 성능을 분석하고 이해할 수 있으며, 동료들과 결과를 쉽게 공유할 수 있습니다.

[Cohere의 이 가이드](https://docs.cohere.com/page/convfinqa-finetuning-wandb)는 파인튜닝 run을 시작하는 전체 예제를 제공합니다. [Cohere API 문서](https://docs.cohere.com/reference/createfinetunedmodel#request.body.settings.wandb)도 참고하실 수 있습니다.

## Cohere 파인튜닝 결과 로그 남기기

Cohere 파인튜닝 로그를 W&B 워크스페이스에 추가하려면:

1. W&B API 키와 W&B `entity`, `project` 이름으로 `WandbConfig`를 생성하세요. W&B API 키는 https://wandb.ai/authorize 에서 확인할 수 있습니다.

2. 이 설정을 `FinetunedModel` 오브젝트에 모델 이름, 데이터셋, 하이퍼파라미터와 함께 전달해 파인튜닝 run을 시작하세요.


    ```python
    from cohere.finetuning import WandbConfig, FinetunedModel

    # W&B 정보를 담은 설정 생성
    wandb_ft_config = WandbConfig(
        api_key="<wandb_api_key>",
        entity="my-entity", # 제공된 API 키와 연결된 유효한 entity여야 합니다
        project="cohere-ft",
    )

    ...  # 데이터셋과 하이퍼파라미터 준비

    # cohere에서 파인튜닝 run 시작
    cmd_r_finetune = co.finetuning.create_finetuned_model(
      request=FinetunedModel(
        name="command-r-ft",
        settings=Settings(
          base_model=...
          dataset_id=...
          hyperparameters=...
          wandb=wandb_ft_config  # 여기에서 W&B 설정을 전달합니다
        ),
      ),
    )
    ```

3. 생성한 W&B project 내에서 모델의 파인튜닝 트레이닝 및 검증 메트릭, 하이퍼파라미터를 확인할 수 있습니다.

    {{< img src="/images/integrations/cohere_ft.png" alt="Cohere fine-tuning dashboard" >}}


## Runs 정리하기

W&B의 runs는 자동으로 정리되며, job 유형, base 모델, 러닝레이트 같은 설정 파라미터 또는 기타 하이퍼파라미터 기준으로 필터링하거나 정렬할 수 있습니다.

추가로, runs의 이름을 변경하거나, 메모를 추가하거나, 태그를 추가해 그룹화할 수도 있습니다.

## 참고 자료

* [Cohere Fine-tuning Example](https://github.com/cohere-ai/notebooks/blob/kkt_ft_cookbooks/notebooks/finetuning/convfinqa_finetuning_wandb.ipynb)