---
title: Rewind a run
description: 되감기
menu:
  default:
    identifier: ko-guides-models-track-runs-rewind
    parent: what-are-runs
---

# run 되감기
{{% alert color="secondary" %}}
run을 되감는 옵션은 비공개 미리 보기 상태입니다. 이 기능에 대한 엑세스를 요청하려면 support@wandb.com으로 W&B 지원팀에 문의하세요.

현재 W&B는 다음을 지원하지 않습니다.
* **로그 되감기**: 새 run 세그먼트에서 로그가 재설정됩니다.
* **시스템 메트릭 되감기**: W&B는 되감기 지점 이후의 새로운 시스템 메트릭만 기록합니다.
* **아티팩트 연결**: W&B는 아티팩트를 해당 아티팩트를 생성하는 소스 run과 연결합니다.
{{% /alert %}}

{{% alert %}}
* run을 되감으려면 [W&B Python SDK](https://pypi.org/project/wandb/) 버전 >= `0.17.1`이 있어야 합니다.
* 단조 증가하는 단계를 사용해야 합니다. [`define_metric()`]({{< relref path="/ref/python/run#define_metric" lang="ko" >}})으로 정의된 비단조 단계를 사용하면 run 기록 및 시스템 메트릭의 필수적인 시간순서가 방해되므로 사용할 수 없습니다.
{{% /alert %}}

run을 되감아 원본 데이터를 잃지 않고 run의 기록을 수정하거나 변경합니다. 또한 run을 되감을 때 해당 시점부터 새로운 데이터를 기록할 수 있습니다. W&B는 새롭게 기록된 기록을 기반으로 되감은 run에 대한 요약 메트릭을 다시 계산합니다. 이는 다음 행동을 의미합니다.
- **기록 잘림**: W&B는 기록을 되감기 지점까지 자르고 새로운 데이터 로깅을 허용합니다.
- **요약 메트릭**: 새롭게 기록된 기록을 기반으로 다시 계산됩니다.
- **설정 보존**: W&B는 원본 설정을 보존하고 새로운 설정을 병합할 수 있습니다.

run을 되감으면 W&B는 run의 상태를 지정된 단계로 재설정하여 원본 데이터를 보존하고 일관된 run ID를 유지합니다. 이는 다음을 의미합니다.

- **run 보관**: W&B는 원본 run을 보관합니다. [**Run Overview**]({{< relref path="./#overview-tab" lang="ko" >}}) 탭에서 run에 엑세스할 수 있습니다.
- **아티팩트 연결**: 아티팩트를 해당 아티팩트를 생성하는 run과 연결합니다.
- **불변 run ID**: 정확한 상태에서 일관된 포킹을 위해 도입되었습니다.
- **불변 run ID 복사**: 향상된 run 관리를 위해 불변 run ID를 복사하는 버튼입니다.

{{% alert title="되감기 및 포킹 호환성" %}}
포킹은 되감기를 보완합니다.

run에서 포크하면 W&B는 특정 시점에서 run의 새로운 분기를 생성하여 다양한 파라미터 또는 모델을 시도합니다.

run을 되감으면 W&B를 통해 run 기록 자체를 수정하거나 변경할 수 있습니다.
{{% /alert %}}

## run 되감기

`resume_from`과 함께 [`wandb.init()`]({{< relref path="/ref/python/init" lang="ko" >}})를 사용하여 run의 기록을 특정 단계로 "되감습니다". 되감으려는 run의 이름과 단계를 지정합니다.

```python
import wandb
import math

# 첫 번째 run을 초기화하고 일부 메트릭을 기록합니다.
# your_project_name 및 your_entity_name으로 바꾸세요!
run1 = wandb.init(project="your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 특정 단계에서 첫 번째 run부터 되감고 200단계부터 메트릭을 기록합니다.
run2 = wandb.init(project="your_project_name", entity="your_entity_name", resume_from=f"{run1.id}?_step=200")

# 새 run에서 계속 기록합니다.
# 처음 몇 단계 동안은 run1에서 메트릭을 그대로 기록합니다.
# 250단계 이후에는 스파이크 패턴을 기록하기 시작합니다.
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i, "step": i})  # 스파이크 없이 run1에서 계속 기록합니다.
    else:
        # 250단계부터 스파이크 행동을 도입합니다.
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 미묘한 스파이크 패턴을 적용합니다.
        run2.log({"metric": subtle_spike, "step": i})
    # 모든 단계에서 새 메트릭을 추가로 기록합니다.
    run2.log({"additional_metric": i * 1.1, "step": i})
run2.finish()
```

## 보관된 run 보기

run을 되감은 후 W&B App UI에서 보관된 run을 탐색할 수 있습니다. 다음 단계에 따라 보관된 run을 봅니다.

1. **Overview 탭에 엑세스**: run 페이지에서 [**Overview 탭**]({{< relref path="./#overview-tab" lang="ko" >}})으로 이동합니다. 이 탭은 run의 세부 정보 및 기록에 대한 포괄적인 보기를 제공합니다.
2. **Forked From 필드 찾기**: **Overview** 탭 내에서 `Forked From` 필드를 찾습니다. 이 필드는 재개의 기록을 캡처합니다. **Forked From** 필드에는 소스 run에 대한 링크가 포함되어 있어 원본 run으로 다시 추적하고 전체 되감기 기록을 이해할 수 있습니다.

`Forked From` 필드를 사용하면 보관된 재개 트리를 쉽게 탐색하고 각 되감기의 순서와 출처에 대한 통찰력을 얻을 수 있습니다.

## 되감은 run에서 포크하기

되감은 run에서 포크하려면 `wandb.init()`에서 [**`fork_from`**]({{< relref path="/guides/models/track/runs/forking" lang="ko" >}}) 인수를 사용하고 소스 run ID와 포크할 소스 run의 단계를 지정합니다.

```python
import wandb

# 특정 단계에서 run을 포크합니다.
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{rewind_run.id}?_step=500",
)

# 새 run에서 계속 기록합니다.
for i in range(500, 1000):
    forked_run.log({"metric": i*3})
forked_run.finish()
```