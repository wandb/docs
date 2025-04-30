---
title: Fork a run
description: W&B run 포크하기
menu:
  default:
    identifier: ko-guides-models-track-runs-forking
    parent: what-are-runs
---

{{% alert color="secondary" %}}
run 포크 기능은 비공개 미리보기로 제공됩니다. 이 기능에 대한 엑세스를 요청하려면 support@wandb.com으로 W&B 지원팀에 문의하십시오.
{{% /alert %}}

기존 W&B run에서 "포크"하려면 [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ko" >}})으로 run을 초기화할 때 `fork_from`을 사용하십시오. run에서 포크하면 W&B는 소스 run의 `run ID` 및 `step`을 사용하여 새 run을 생성합니다.

run을 포크하면 원래 run에 영향을 주지 않고 실험의 특정 시점에서 다른 파라미터 또는 Models를 탐색할 수 있습니다.

{{% alert %}}
* run을 포크하려면 [`wandb`](https://pypi.org/project/wandb/) SDK 버전 >= 0.16.5가 필요합니다
* run을 포크하려면 단조 증가하는 steps이 필요합니다. [`define_metric()`]({{< relref path="/ref/python/run#define_metric" lang="ko" >}})으로 정의된 비단조 steps을 사용하여 포크 지점을 설정하면 run 기록 및 시스템 메트릭의 필수적인 시간순서가 손상되므로 사용할 수 없습니다.
{{% /alert %}}

## 포크된 run 시작

run을 포크하려면 [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ko" >}})에서 `fork_from` 인수를 사용하고 포크할 소스 `run ID`와 소스 run의 `step`을 지정하십시오.

```python
import wandb

# 나중에 포크할 run을 초기화합니다
original_run = wandb.init(project="your_project_name", entity="your_entity_name")
# ... 트레이닝 또는 로깅 수행 ...
original_run.finish()

# 특정 step에서 run을 포크합니다
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{original_run.id}?_step=200",
)
```

### 변경 불가능한 run ID 사용

특정 run에 대한 일관되고 변경되지 않는 참조를 보장하려면 변경 불가능한 run ID를 사용하십시오. 사용자 인터페이스에서 변경 불가능한 run ID를 얻으려면 다음 단계를 따르십시오.

1. **Overview 탭에 엑세스:** 소스 run 페이지의 [**Overview 탭**]({{< relref path="./#overview-tab" lang="ko" >}})으로 이동합니다.

2. **변경 불가능한 Run ID 복사:** **Overview** 탭의 오른쪽 상단에 있는 `...` 메뉴(세 개의 점)를 클릭합니다. 드롭다운 메뉴에서 `Copy Immutable Run ID` 옵션을 선택합니다.

이러한 단계를 따르면 run에 대한 안정적이고 변경되지 않는 참조를 갖게 되어 run을 포크하는 데 사용할 수 있습니다.

## 포크된 run에서 계속하기
포크된 run을 초기화한 후 새 run에 계속 로그할 수 있습니다. 연속성을 위해 동일한 메트릭을 로그하고 새 메트릭을 도입할 수 있습니다.

예를 들어 다음 코드 예제에서는 먼저 run을 포크한 다음 트레이닝 step 200부터 포크된 run에 메트릭을 로그하는 방법을 보여줍니다.

```python
import wandb
import math

# 첫 번째 run을 초기화하고 일부 메트릭을 로그합니다
run1 = wandb.init("your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 특정 step에서 첫 번째 run에서 포크하고 step 200부터 메트릭을 로그합니다
run2 = wandb.init(
    "your_project_name", entity="your_entity_name", fork_from=f"{run1.id}?_step=200"
)

# 새 run에서 계속 로깅합니다
# 처음 몇 steps 동안은 run1에서 메트릭을 그대로 로깅합니다
# Step 250 이후에는 스파이크 패턴 로깅을 시작합니다
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i})  # 스파이크 없이 run1에서 계속 로깅합니다
    else:
        # Step 250부터 스파이크 행동을 도입합니다
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 미묘한 스파이크 패턴을 적용합니다
        run2.log({"metric": subtle_spike})
    # 모든 steps에서 새 메트릭을 추가로 로깅합니다
    run2.log({"additional_metric": i * 1.1})
run2.finish()
```

{{% alert title=" 되감기 및 포크 호환성" %}}
포크는 run을 관리하고 실험하는 데 더 많은 유연성을 제공하여 [`되감기`]({{< relref path="/guides/models/track/runs/rewind" lang="ko" >}})를 보완합니다.

run에서 포크하면 W&B는 특정 시점에서 run에서 새 분기를 생성하여 다른 파라미터 또는 Models를 시도합니다.

run을 되감으면 W&B를 통해 run 기록 자체를 수정하거나 변경할 수 있습니다.
{{% /alert %}}
