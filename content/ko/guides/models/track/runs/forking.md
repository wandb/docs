---
title: Fork a run
description: W&B run 포크
menu:
  default:
    identifier: ko-guides-models-track-runs-forking
    parent: what-are-runs
---

{{% alert color="secondary" %}}
run 을 포크하는 기능은 비공개 미리 보기로 제공됩니다. 이 기능에 대한 엑세스 권한을 요청하려면 support@wandb.com 으로 W&B 지원팀에 문의하십시오.
{{% /alert %}}

기존 W&B run 에서 "포크"하려면 [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ko" >}}) 으로 run 을 초기화할 때 `fork_from` 을 사용하십시오. run 에서 포크하면 W&B 는 소스 run 의 `run ID` 와 `step` 을 사용하여 새 run 을 생성합니다.

run 을 포크하면 원래 run 에 영향을 주지 않고 실험의 특정 시점에서 다른 파라미터 또는 model 을 탐색할 수 있습니다.

{{% alert %}}
* run 을 포크하려면 [`wandb`](https://pypi.org/project/wandb/) SDK 버전 >= 0.16.5 가 필요합니다.
* run 을 포크하려면 단조 증가하는 step 이 필요합니다. [`define_metric()`]({{< relref path="/ref/python/run#define_metric" lang="ko" >}}) 으로 정의된 비단조 step 을 사용하여 포크 지점을 설정하면 run 기록 및 시스템 메트릭의 필수적인 시간순서가 중단되므로 사용할 수 없습니다.
{{% /alert %}}

## 포크된 run 시작

run 을 포크하려면 [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ko" >}}) 에서 `fork_from` 인수를 사용하고 포크할 소스 `run ID` 와 소스 run 의 `step` 을 지정합니다.

```python
import wandb

# 나중에 포크할 run 을 초기화합니다.
original_run = wandb.init(project="your_project_name", entity="your_entity_name")
# ... 트레이닝 또는 로깅 수행 ...
original_run.finish()

# 특정 step 에서 run 을 포크합니다.
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{original_run.id}?_step=200",
)
```

### 변경 불가능한 run ID 사용

변경 불가능한 run ID 를 사용하여 특정 run 에 대한 일관되고 변경되지 않는 참조를 확인합니다. 다음 단계에 따라 사용자 인터페이스에서 변경 불가능한 run ID 를 가져옵니다.

1. **Overview 탭 에 엑세스:** 소스 run 의 페이지에서 [**Overview 탭**]({{< relref path="./#overview-tab" lang="ko" >}}) 으로 이동합니다.

2. **변경 불가능한 Run ID 복사:** **Overview** 탭 의 오른쪽 상단에 있는 `...` 메뉴(세 개의 점)를 클릭합니다. 드롭다운 메뉴에서 `Copy Immutable Run ID` 옵션을 선택합니다.

이러한 단계를 따르면 run 에 대한 안정적이고 변경되지 않는 참조를 얻을 수 있으며, 이는 run 을 포크하는 데 사용할 수 있습니다.

## 포크된 run 부터 계속

포크된 run 을 초기화한 후 새 run 에 계속 로깅할 수 있습니다. 연속성을 위해 동일한 메트릭을 로깅하고 새 메트릭을 도입할 수 있습니다.

예를 들어 다음 코드 예제는 먼저 run 을 포크한 다음 트레이닝 step 200부터 포크된 run 에 메트릭을 로깅하는 방법을 보여줍니다.

```python
import wandb
import math

# 첫 번째 run 을 초기화하고 일부 메트릭을 로깅합니다.
run1 = wandb.init("your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 특정 step 에서 첫 번째 run 에서 포크하고 step 200부터 메트릭을 로깅합니다.
run2 = wandb.init(
    "your_project_name", entity="your_entity_name", fork_from=f"{run1.id}?_step=200"
)

# 새 run 에서 계속 로깅합니다.
# 처음 몇 step 동안은 run1에서 메트릭을 그대로 로깅합니다.
# step 250 이후에는 스파이크 패턴을 로깅하기 시작합니다.
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i})  # 스파이크 없이 run1부터 계속 로깅합니다.
    else:
        # step 250부터 스파이크 동작을 도입합니다.
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 미묘한 스파이크 패턴을 적용합니다.
        run2.log({"metric": subtle_spike})
    # 또한 모든 step 에서 새 메트릭을 로깅합니다.
    run2.log({"additional_metric": i * 1.1})
run2.finish()
```

{{% alert title=" 되감기 및 포크 호환성" %}}
포크는 run 을 관리하고 실험하는 데 더 많은 유연성을 제공하여 [`rewind`]({{< relref path="/guides/models/track/runs/rewind" lang="ko" >}}) 를 보완합니다.

run 에서 포크하면 W&B 는 특정 시점에서 run 에서 새 분기를 만들어 다른 파라미터 또는 model 을 시도합니다.

run 을 되감으면 W&B 를 통해 run 기록 자체를 수정하거나 변경할 수 있습니다.
{{% /alert %}}
