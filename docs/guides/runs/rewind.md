---
title: Rewind a run
description: 리와인드
displayed_sidebar: default
---

# Run 되돌리기

:::caution
run을 되돌리는 기능은 비공개 프리뷰 상태입니다. 이 기능에 엑세스하려면 support@wandb.com으로 W&B 지원 팀에 문의하세요.

현재 W&B에서는 다음을 지원하지 않습니다:
* **로그 되돌리기**: 로그는 새로운 run 구간에서 초기화됩니다.
* **시스템 메트릭 되돌리기**: 되돌리기 지점 이후의 새로운 시스템 메트릭만 로그에 기록됩니다.
* **아티팩트 연관**: 아티팩트는 이를 생성한 소스 run과 연관됩니다.
:::

:::info
* run을 되돌리려면 [W&B Python SDK](https://pypi.org/project/wandb/) 버전이 `0.17.1` 이상이어야 합니다.
* 순차적으로 증가하는 단계를 사용해야 합니다. [`define_metric()`](/ref/python/run#define_metric)로 정의된 비순차적인 단계는 사용할 수 없습니다. 이는 run 기록 및 시스템 메트릭의 필요한 연대순서를 방해할 수 있기 때문입니다.
:::

run을 되돌려 run의 기록을 원본 데이터를 잃지 않고 수정하거나 변경하세요. 또한, run을 되돌리면 그 시점부터 새로운 데이터를 로그할 수 있습니다. 되돌린 run의 요약 메트릭은 새로 로그된 기록을 바탕으로 다시 계산됩니다. 이는 다음과 같은 행동을 의미합니다:
- **기록 잘라내기**: 기록은 되돌리기 지점까지 잘라내어, 새로운 데이터 로그가 가능해집니다.
- **요약 메트릭**: 새로 로그된 기록을 기반으로 다시 계산됩니다.
- **설정 보존**: 원본 설정은 보존되며 새로운 설정과 병합될 수 있습니다.

run을 되돌릴 때, W&B는 지정된 단계로 run의 상태를 초기화하고 원본 데이터를 보존하며 일관된 run ID를 유지합니다. 이는 다음을 의미합니다:

- **run 아카이브**: 원본 run은 아카이브되어 [**Run Overview**](/guides/app/pages/run-page#overview-tab) 탭에서 엑세스할 수 있습니다.
- **아티팩트 연관**: 아티팩트는 이를 생성한 run과 연관됩니다.
- **불변의 run ID**: 정확한 상태에서 일관된 가지치기를 위해 도입되었습니다.
- **불변의 run ID 복사**: 향상된 run 관리를 위해 불변의 run ID를 복사할 수 있는 버튼이 제공됩니다.

:::tip 되돌리기와 가지치기 호환성
가지치기는 run의 관리 및 실험에 더 유연성을 제공하여 [`rewind`](/guides/runs/rewind)를 보완합니다.

run에서 가지치기를 할 때, W&B는 특정 지점에서 run의 새로운 분기를 생성하여 다른 파라미터 또는 모델을 시도할 수 있게 합니다.

run을 되돌릴 때, W&B는 run 기록 자체를 수정하거나 변경할 수 있도록 합니다.
:::

## Run 되돌리기

`resume_from`을 [`wandb.init()`](/ref/python/init)와 함께 사용하여 run의 기록을 특정 단계로 "되돌리"십시오. 되돌릴 run의 이름과 되돌리고자 하는 단계를 지정하세요:

```python
import wandb
import math

# 첫 번째 run을 초기화하고 몇 가지 메트릭을 로그합니다.
# your_project_name 및 your_entity_name로 변경하세요!
run1 = wandb.init(project="your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 특정 단계에서 첫 번째 run을 되돌리고, 200 단계부터 메트릭을 로그합니다.
run2 = wandb.init(project="your_project_name", entity="your_entity_name", resume_from=f"{run1.id}?_step=200")

# 새로운 run에서 로그를 계속합니다.
# 처음 몇 단계에서는 run1의 메트릭을 서서히 로그합니다.
# 250 단계 이후부터는 선명한 패턴을 로그하기 시작합니다.
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i, "step": i})  # run1에서 선명한 패턴 없이 로그를 계속합니다.
    else:
        # 250 단계에서 선명한 행동을 도입합니다.
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 미세한 선명한 패턴을 적용합니다.
        run2.log({"metric": subtle_spike, "step": i})
    # 모든 단계에서 새로운 메트릭을 추가로 로그합니다.
    run2.log({"additional_metric": i * 1.1, "step": i})
run2.finish()
```

## 아카이브된 run 보기

run을 되돌린 후, W&B App UI를 사용하여 아카이브된 run을 탐색할 수 있습니다. 아카이브된 run을 확인하려면 다음 단계를 따르세요:

1. **Overview 탭 엑세스:** run의 페이지에서 [**Overview 탭**](/guides/app/pages/run-page#overview-tab)으로 이동합니다. 이 탭은 run의 세부 정보와 기록에 대한 포괄적인 보기를 제공합니다.
2. **Forked From 필드 찾기:** **Overview** 탭 내에 `Forked From` 필드를 찾습니다. 이 필드는 재개 이력을 캡처합니다. **Forked From** 필드는 소스 run으로의 링크를 포함하여, 원본 run까지 추적하고 전체 되돌리기 이력을 이해할 수 있도록 합니다.

`Forked From` 필드를 사용하여 아카이브된 재개의 트리를 쉽게 탐색하고 각 되돌리기의 순서 및 기원을 파악할 수 있습니다.

## 되돌린 run에서 가지치기

되돌린 run에서 가지치기하려면, `wandb.init()`에서 [**`fork_from`**](/guides/runs/forking) 인수를 사용하고 소스 run ID와 가지치기할 단계 ID를 지정하세요:

```python
import wandb

# 특정 단계에서 run을 가지치기합니다.
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{rewind_run.id}?_step=500",
)

# 새로운 run에서 로그를 계속합니다.
for i in range(500, 1000):
    forked_run.log({"metric": i*3})
forked_run.finish()
```