---
title: Hugging Face Accelerate
description: 대규모 트레이닝과 추론을 쉽고 효율적이며 유연하게 만들어 드립니다
menu:
  default:
    identifier: ko-guides-integrations-accelerate
    parent: integrations
weight: 140
---

Hugging Face Accelerate는 동일한 PyTorch 코드를 어떤 분산 설정에서도 실행할 수 있도록 해주는 라이브러리로, 대규모 모델 트레이닝과 추론을 간소화해줍니다.

Accelerate에는 W&B Tracker가 포함되어 있으며, 아래에서 사용하는 방법을 안내합니다. 또한 [Hugging Face의 Accelerate Trackers에 대해 더 알아보기](https://huggingface.co/docs/accelerate/main/en/usage_guides/tracking)에서 자세한 내용을 확인할 수 있습니다.

## Accelerate로 로깅 시작하기

Accelerate와 W&B를 시작하려면 아래의 예시 코드를 참고하세요:

```python
from accelerate import Accelerator

# Accelerator 오브젝트에게 wandb로 로그를 남기도록 지정합니다
accelerator = Accelerator(log_with="wandb")

# wandb run을 초기화하고, wandb 파라미터 및 관련 config 정보를 전달합니다
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2}
    init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )

...

# `accelerator.log`를 호출하여 wandb에 로그를 남깁니다. `step`은 선택 사항입니다
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=global_step)


# wandb tracker가 올바르게 종료되도록 합니다
accelerator.end_training()
```

더 설명하자면, 다음을 수행해야 합니다:
1. Accelerator 클래스를 초기화할 때 `log_with="wandb"`를 전달합니다.
2. [`init_trackers`](https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.init_trackers) 메소드를 호출하고 아래를 전달합니다:
   - `project_name`을 통해 프로젝트 이름
   - [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}})에 전달할 파라미터는 `init_kwargs`에 중첩된 dict로 전달
   - wandb run에 기록하고자 하는 기타 실험 config 정보는 `config`로 전달
3. `.log` 메소드를 사용하여 Weights & Biases에 로그를 남깁니다. `step` 인수는 선택 사항입니다.
4. 트레이닝이 종료되면 `.end_training`을 호출합니다.

## W&B tracker 엑세스하기

W&B tracker에 엑세스하려면 `Accelerator.get_tracker()` 메소드를 사용하세요. 트래커의 `.name` 속성에 해당하는 문자열을 전달하면, `main` 프로세스에서 해당 tracker를 반환합니다.

```python
wandb_tracker = accelerator.get_tracker("wandb")
```

이후에는 평소처럼 wandb의 run 오브젝트와 상호작용할 수 있습니다:

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```

{{% alert color="secondary" %}}
Accelerate에 내장된 트래커들은 자동으로 올바른 프로세스에서 실행됩니다. 메인 프로세스에서만 실행되어야 하는 경우라면, 자동으로 그렇게 처리됩니다.

Accelerate의 래핑을 완전히 제거하고 싶다면 아래와 같이 동일한 결과를 얻을 수 있습니다:

```python
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
with accelerator.on_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```
{{% /alert %}}

## Accelerate 관련 아티클
아래는 Accelerate와 관련된 아티클입니다.

<details>

<summary>HuggingFace Accelerate Super Charged With W&B</summary>

* 이 아티클에서는 HuggingFace Accelerate가 제공하는 기능과 얼마나 쉽게 분산 트레이닝과 평가를 수행할 수 있는지, 그리고 결과를 W&B로 로깅하는 방법을 살펴봅니다.

[Hugging Face Accelerate Super Charged with W&B 리포트 읽기](https://wandb.ai/gladiator/HF%20Accelerate%20+%20W&B/reports/Hugging-Face-Accelerate-Super-Charged-with-Weights-Biases--VmlldzoyNzk3MDUx?utm_source=docs&utm_medium=docs&utm_campaign=accelerate-docs).
</details>
<br /><br />