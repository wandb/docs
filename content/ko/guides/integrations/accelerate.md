---
title: Hugging Face Accelerate
description: 간단하고 효율적이며 적응 가능한 대규모 트레이닝 및 추론
menu:
  default:
    identifier: ko-guides-integrations-accelerate
    parent: integrations
weight: 140
---

Hugging Face Accelerate는 분산 설정 전반에서 동일한 PyTorch 코드를 실행하여 대규모 모델 트레이닝 및 추론을 간소화하는 라이브러리입니다.

Accelerate에는 아래에서 사용하는 방법을 보여주는 Weights & Biases Tracker가 포함되어 있습니다. Accelerate Trackers에 대한 자세한 내용은 **[여기에서 해당 문서를 참조하십시오](https://huggingface.co/docs/accelerate/main/en/usage_guides/tracking)**

## Accelerate로 로깅 시작하기

Accelerate 및 Weights & Biases를 시작하려면 아래의 의사 코드를 따르십시오.

```python
from accelerate import Accelerator

# Accelerator 오브젝트에게 wandb 로 로그하도록 지시
accelerator = Accelerator(log_with="wandb")

# wandb run을 초기화하고 wandb 파라미터 및 모든 설정 정보 전달
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2}
    init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )

...

# `accelerator.log`를 호출하여 wandb에 로그, `step`은 선택 사항
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=global_step)


# wandb tracker가 올바르게 완료되었는지 확인
accelerator.end_training()
```

자세히 설명하자면 다음이 필요합니다.
1. Accelerator 클래스를 초기화할 때 `log_with="wandb"`를 전달합니다.
2. [`init_trackers`](https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.init_trackers) 메소드를 호출하고 다음을 전달합니다.
- `project_name`을 통해 프로젝트 이름
- 중첩된 dict을 통해 [`wandb.init`]({{< relref path="/ref/python/init" lang="ko" >}})에 전달할 파라미터는 `init_kwargs`에 전달
- `config`를 통해 wandb run에 로그할 다른 모든 실험 설정 정보
3. `.log` 메소드를 사용하여 Weights & Biases에 로그합니다. `step` 인수는 선택 사항입니다.
4. 트레이닝이 완료되면 `.end_training`을 호출합니다.

## W&B tracker 액세스

W&B tracker에 액세스하려면 `Accelerator.get_tracker()` 메소드를 사용하십시오. tracker의 `.name` 속성에 해당하는 문자열을 전달하면 `main` 프로세스에서 tracker가 반환됩니다.

```python
wandb_tracker = accelerator.get_tracker("wandb")

```
거기에서 평소처럼 wandb의 run 오브젝트와 상호 작용할 수 있습니다.

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```

{{% alert color="secondary" %}}
Accelerate에 내장된 Trackers는 올바른 프로세스에서 자동으로 실행되므로, tracker가 main 프로세스에서만 실행되도록 설계된 경우 자동으로 실행됩니다.

Accelerate의 래핑을 완전히 제거하려면 다음과 같이 동일한 결과를 얻을 수 있습니다.

```python
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
with accelerator.on_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```
{{% /alert %}}

## Accelerate 관련 기사
다음은 즐겨보실 만한 Accelerate 관련 기사입니다.

<details>

<summary>Weights & Biases로 강화된 HuggingFace Accelerate</summary>

* 이 기사에서는 HuggingFace Accelerate가 제공하는 기능과 결과를 Weights & Biases에 로깅하면서 분산 트레이닝 및 평가를 수행하는 것이 얼마나 간단한지 살펴보겠습니다.

전체 리포트는 [여기](https://wandb.ai/gladiator/HF%20Accelerate%20+%20W&B/reports/Hugging-Face-Accelerate-Super-Charged-with-Weights-Biases--VmlldzoyNzk3MDUx?utm_source=docs&utm_medium=docs&utm_campaign=accelerate-docs)에서 읽어보세요.
</details>
<br /><br />
