---
title: Hugging Face Accelerate
description: 대규모 트레이닝과 추론을 간단하고 효율적이며 적응 가능하게 만드는 방법
slug: /guides/integrations/accelerate
displayed_sidebar: default
---

Accelerate는 PyTorch 코드를 네 줄의 코드만 추가하여 어떤 분산 설정에서든 실행할 수 있게 하는 라이브러리로, 대규모의 트레이닝과 추론을 간단하고 효율적으로, 그리고 적응할 수 있게 만들어 줍니다.

Accelerate에는 Weights & Biases Tracker가 포함되어 있으며, 아래에 사용하는 방법을 보여드립니다. 또한 **[이곳의 문서](https://huggingface.co/docs/accelerate/main/en/usage_guides/tracking)**에서 Accelerate Trackers에 대해 더 읽어볼 수 있습니다.

## Accelerate로 로깅 시작하기

Accelerate와 Weights & Biases로 시작하려면 아래의 의사 코드를 따라 할 수 있습니다:

```python
from accelerate import Accelerator

# Accelerator 오브젝트에 wandb로 로그하도록 지시합니다
accelerator = Accelerator(log_with="wandb")

# wandb run을 초기화하고, wandb 파라미터와 설정 정보를 전달합니다
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2}
    init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )

...

# `accelerator.log`를 호출하여 wandb에 로그합니다, `step`은 선택 사항입니다
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=global_step)


# wandb tracker가 올바르게 종료되도록 확인합니다
accelerator.end_training()
```

추가 설명으로, 다음과 같이 해야 합니다:
1. Accelerator 클래스를 초기화할 때 `log_with="wandb"`를 전달합니다
2. [`init_trackers`](https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.init_trackers) 메소드를 호출하고 다음을 전달합니다:
- `project_name`을 통해 프로젝트 이름을 전달합니다
- [`wandb.init`](/ref/python/init)에 전달할 파라미터를 `init_kwargs`라는 중첩된 dict로 전달합니다
- wandb run에 로그하고자 하는 기타 실험 설정 정보를 `config`로 전달합니다
3. Weights & Biases에 로그하기 위해 `.log` 메소드를 사용합니다; `step` 인수는 선택 사항입니다
4. 트레이닝이 끝났을 때 `.end_training`을 호출합니다

## Accelerates' 내부 W&B 트래커 엑세스하기

Accelerator.get_tracker() 메소드를 사용하여 wandb 트래커에 빠르게 엑세스할 수 있습니다. 트래커의 .name 속성에 해당하는 문자열을 전달하면 메인 프로세스에서 해당 트래커를 반환합니다.

```python
wandb_tracker = accelerator.get_tracker("wandb")

```
여기에서 wandb의 run 오브젝트와 평소처럼 상호작용할 수 있습니다:

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```

:::caution
Accelerate에 내장된 트래커는 자동으로 올바른 프로세스에서 실행됩니다. 따라서 트래커가 메인 프로세스에서만 실행되어야 한다면 자동으로 그렇게 실행됩니다.

Accelerate의 래핑을 완전히 제거하고 싶다면, 다음과 같은 방법으로 동일한 결과를 얻을 수 있습니다:

```
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
with accelerator.on_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```
:::

## Accelerate 기사
아래는 Accelerate 기사로, 당신이 즐길 만한 내용입니다

<details>

<summary>HuggingFace Accelerate Super Charged With Weights & Biases</summary>

* 이 기사에서는 HuggingFace Accelerate가 제공하는 것과, 분산 트레이닝과 평가를 수행하면서 Weights & Biases에 결과를 로깅하는 방법의 간단함에 대해 살펴봅니다

전체 리포트는 [여기](https://wandb.ai/gladiator/HF%20Accelerate%20+%20W&B/reports/Hugging-Face-Accelerate-Super-Charged-with-Weights-Biases--VmlldzoyNzk3MDUx?utm_source=docs&utm_medium=docs&utm_campaign=accelerate-docs)에서 읽어보세요.
</details>