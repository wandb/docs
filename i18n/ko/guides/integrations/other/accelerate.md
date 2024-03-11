---
description: Training and inference at scale made simple, efficient and adaptable
slug: /guides/integrations/accelerate
displayed_sidebar: default
---

# Hugging Face Accelerate

Accelerate는 단 4줄의 코드를 추가함으로써 동일한 PyTorch 코드를 어떠한 분산 구성에서도 실행할 수 있게 해주는 라이브러리로, 규모에 맞게 트레이닝과 추론을 간단하고 효율적이며 적응성 있게 만듭니다.

Accelerate에는 Weights & Biases Tracker가 포함되어 있으며, 아래에서 그 사용 방법을 보여드립니다. Accelerate Trackers에 대해 더 알아보려면 **[여기서 문서를 확인하세요](https://huggingface.co/docs/accelerate/main/en/usage_guides/tracking)**

## Accelerate로 로깅 시작하기

Accelerate와 Weights & Biases를 시작하려면 아래의 가이드 코드를 따라하세요:

```python
from accelerate import Accelerator

# Accelerator 객체에게 wandb로 로그하도록 지정
accelerator = Accelerator(log_with="wandb")

# wandb run을 초기화하고, wandb 파라미터와 모든 설정 정보를 전달
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2}
    init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )

...

# `accelerator.log`를 호출하여 wandb에 로그, `step`은 선택 사항
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=global_step)


# wandb 트래커가 올바르게 종료되었는지 확인
accelerator.end_training()
```

더 자세히 설명하자면, 다음과 같이 해야 합니다:
1. Accelerator 클래스를 초기화할 때 `log_with="wandb"`를 전달합니다.
2. [`init_trackers`](https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.init_trackers) 메소드를 호출하고 다음을 전달합니다:
- `project_name`을 통해 프로젝트 이름
- `init_kwargs`에 중첩된 dict를 통해 [`wandb.init`](https://docs.wandb.ai/ref/python/init)에 전달할 파라미터
- `config`를 통해 wandb run에 로그할 다른 실험 설정 정보
3. Weigths & Biases에 로그하기 위해 `.log` 메소드를 사용합니다; `step` 인수는 선택 사항입니다
4. 트레이닝이 끝났을 때 `.end_training`을 호출합니다

## Accelerates의 내부 W&B 트래커 접근하기

Accelerator.get_tracker() 메소드를 사용하면 wandb 트래커에 빠르게 접근할 수 있습니다. 트래커의 `.name` 속성에 해당하는 문자열을 전달하면 메인 프로세스에서 해당 트래커를 반환합니다.

```python
wandb_tracker = accelerator.get_tracker("wandb")

```
거기에서 평소처럼 wandb의 run 오브젝트와 상호작용할 수 있습니다:

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```

:::caution
Accelerate에 내장된 트래커는 올바른 프로세스에서 자동으로 실행되므로, 트래커가 메인 프로세스에서만 실행되어야 한다면 자동으로 그렇게 됩니다.

Accelerate의 래핑을 완전히 제거하려면, 다음과 같이 할 수 있습니다:

```
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
with accelerator.on_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```
:::

## Accelerate 기사
아래는 Accelerate 관련 기사입니다

<details>

<summary>HuggingFace Accelerate가 Weights & Biases로 강화됨</summary>

* 이 기사에서는 HuggingFace Accelerate가 제공하는 것과 분산 트레이닝과 평가를 수행하는 것이 얼마나 간단한지, 그리고 Weights & Biases에 결과를 로깅하는 방법을 살펴봅니다.

전체 리포트를 [여기서 읽어보세요](https://wandb.ai/gladiator/HF%20Accelerate%20+%20W&B/reports/Hugging-Face-Accelerate-Super-Charged-with-Weights-Biases--VmlldzoyNzk3MDUx?utm_source=docs&utm_medium=docs&utm_campaign=accelerate-docs).
</details>