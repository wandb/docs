---
description: Training and inference at scale made simple, efficient and adaptable
slug: /guides/integrations/accelerate
displayed_sidebar: default
---

# Hugging Face Accelerate

Accelerate는 단 네 줄의 코드만 추가하면 동일한 PyTorch 코드를 어떤 분산 구성에서도 실행할 수 있게 해주는 라이브러리로, 규모에 맞는 학습 및 추론을 간단하고 효율적이며 적응성 있게 만듭니다.

Accelerate에는 Weights & Biases 추적기가 포함되어 있으며, 아래에서 사용 방법을 보여드립니다. Accelerate 추적기에 대해 더 자세히 알아보려면 **[여기서 그들의 문서를 참조하세요](https://huggingface.co/docs/accelerate/main/en/usage_guides/tracking)**

## Accelerate로 로깅 시작하기

Accelerate와 Weights & Biases를 시작하려면 아래의 가이드 코드를 따르세요:

```python
from accelerate import Accelerator

# Accelerator 객체에 wandb로 로깅하도록 지시
accelerator = Accelerator(log_with="wandb")

# wandb 실행을 초기화하고, wandb 파라미터와 구성 정보를 전달
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2}
    init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )

...

# `accelerator.log`을 호출하여 wandb에 로그, `step`은 선택 사항
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=global_step)


# wandb 추적기가 올바르게 종료되도록 합니다
accelerator.end_training()
```

더 자세히 설명하자면, 다음을 수행해야 합니다:
1. Accelerator 클래스를 초기화할 때 `log_with="wandb"`를 전달합니다.
2. [`init_trackers`](https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.init_trackers) 메서드를 호출하고 다음을 전달합니다:
- `project_name`을 통해 프로젝트 이름
- [`wandb.init`](https://docs.wandb.ai/ref/python/init)에 전달하고자 하는 파라미터를 `init_kwargs`에 중첩된 dict로
- wandb 실행에 로그하고자 하는 다른 실험 구성 정보를 `config`를 통해
3. `.log` 메서드를 사용하여 Weights & Biases에 로그합니다; `step` 인수는 선택 사항입니다.
4. 학습이 끝났을 때 `.end_training`을 호출합니다.

## Accelerates의 내부 W&B 추적기 엑세스

Accelerator.get_tracker() 메서드를 사용하여 wandb 추적기에 빠르게 엑세스할 수 있습니다. 추적기의 `.name` 속성에 해당하는 문자열을 전달하면 메인 프로세스에서 해당 추적기를 반환합니다.

```python
wandb_tracker = accelerator.get_tracker("wandb")

```
여기서부터는 평소처럼 wandb의 run 개체와 정상적으로 상호 작용할 수 있습니다:

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```

:::caution
Accelerate에 내장된 추적기는 올바른 프로세스에서 자동으로 실행되므로, 주 프로세스에서만 실행되어야 하는 추적기는 자동으로 그렇게 할 것입니다.

Accelerate의 래핑을 완전히 제거하고 싶다면, 다음과 같이 동일한 결과를 얻을 수 있습니다:

```
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
with accelerator.on_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```
:::

## Accelerate 기사
아래는 Accelerate에 관한 기사입니다

<details>

<summary>HuggingFace Accelerate Super Charged With Weights & Biases</summary>

* 이 기사에서는 HuggingFace Accelerate가 제공하는 것과 분산 학습 및 평가를 수행하는 것이 얼마나 간단한지, 그리고 Weights & Biases에 결과를 로깅하는 방법을 살펴보겠습니다.

전체 리포트를 [여기서](https://wandb.ai/gladiator/HF%20Accelerate%20+%20W&B/reports/Hugging-Face-Accelerate-Super-Charged-with-Weights-Biases--VmlldzoyNzk3MDUx?utm_source=docs&utm_medium=docs&utm_campaign=accelerate-docs) 읽어보세요.
</details>