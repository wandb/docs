---
title: Fork a run
description: W&B run 포크하기
displayed_sidebar: default
---

:::caution
run을 포크할 수 있는 기능은 사전 검토 중에 있습니다. 이 기능을 엑세스하려면 support@wandb.com으로 W&B 지원팀에 연락하세요.
:::

`fork_from`을 사용하여 [`wandb.init()`](../../ref/python/init.md)으로 run을 초기화할 때 기존의 W&B run을 "포크"합니다. run을 포크하면 W&B는 소스 run의 `run ID`와 `step`을 사용하여 새 run을 생성합니다.

run을 포크하면 실험의 특정 지점에서 원래 run에 영향을 주지 않고 다른 파라미터나 모델을 탐색할 수 있습니다.

:::info
* run을 포크하려면 [`wandb`](https://pypi.org/project/wandb/) SDK 버전 >= 0.16.5가 필요합니다.
* run을 포크하려면 단조롭게 증가하는 step이 필요합니다. [`define_metric()`](/ref/python/run#define_metric)으로 정의된 비단조의 step을 포크 지점으로 설정할 수 없습니다. 이는 run 히스토리와 시스템 메트릭의 필수적인 시간순서를 방해하기 때문입니다.
:::


## 포크된 run 시작하기

run을 포크하려면 [`wandb.init()`](../../ref/python/init.md)에서 `fork_from` 인수를 사용하고 포크할 소스 run의 `run ID`와 `step`을 지정하세요:

```python
import wandb

# 나중에 포크할 run을 초기화합니다
original_run = wandb.init(project="your_project_name", entity="your_entity_name")
# ... 트레이닝이나 로그 작업 수행 ...
original_run.finish()

# 특정 step에서 run을 포크합니다
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{original_run.id}?_step=200",
)
```

### 불변의 run ID 사용하기

특정 run에 대한 일관되고 변하지 않는 참조를 보장하려면 불변의 run ID를 사용하세요. 사용자 인터페이스에서 불변의 run ID를 얻기 위한 다음 단계들을 따라주세요:

1. **Overview 탭 엑세스:** 소스 run 페이지에서 [**Overview 탭**](/guides/app/pages/run-page#overview-tab)으로 이동합니다.

2. **불변의 Run ID 복사**: **Overview** 탭의 오른쪽 상단에 있는 `...` 메뉴(세 개의 점)를 클릭합니다. 드롭다운 메뉴에서 `Copy Immutable Run ID` 옵션을 선택합니다.

이 단계들을 따르면 run을 포크할 수 있는 안정적이고 변하지 않는 참조를 갖게 됩니다.

## 포크된 run에서 계속하기
포크된 run을 초기화한 후 새 run에 로그를 계속할 수 있습니다. 지속성을 위해 동일한 메트릭을 로그할 수 있으며 새 메트릭을 도입할 수도 있습니다.

예를 들어, 다음 코드 예제는 run을 먼저 포크한 후 트레이닝 step 200에서 시작하여 포크된 run에 메트릭을 로그하는 방법을 보여줍니다.

```python
import wandb
import math

# 첫 번째 run을 초기화하고 메트릭을 로그합니다
run1 = wandb.init("your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 특정 step에서 첫 번째 run을 포크하고 step 200에서 시작하여 메트릭을 로그합니다
run2 = wandb.init(
    "your_project_name", entity="your_entity_name", fork_from=f"{run1.id}?_step=200"
)

# 새 run에 로그를 계속합니다
# 첫 번째 몇 가지 step에서는 run1에서의 메트릭을 그대로 로그합니다
# step 250 이후에는 스파이크 패턴을 기록하기 시작합니다
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i})  # 스파이크 없이 run1에서 계속 로그
    else:
        # step 250부터 스파이크 행동을 도입합니다
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 미세한 스파이크 패턴 적용
        run2.log({"metric": subtle_spike})
    # 추가로 모든 step에서 새로운 메트릭도 로그
    run2.log({"additional_metric": i * 1.1})
run2.finish()
```

:::tip 리와인드 및 포크 호환성
포크는 run을 관리하고 실험하는 데 유연성을 제공하는 [`rewind`](/guides/runs/rewind)와 보완적입니다.

run을 포크할 때 W&B는 다른 파라미터나 모델을 시도하기 위해 run의 특정 지점에서 새 분기를 생성합니다.

run을 리와인드할 때는 W&B가 run 히스토리 자체를 수정하거나 변경할 수 있도록 지원합니다.
:::
