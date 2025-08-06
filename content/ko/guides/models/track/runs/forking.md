---
title: run 포크하기
description: W&B run 포크하기
menu:
  default:
    identifier: ko-guides-models-track-runs-forking
    parent: what-are-runs
---

{{% alert color="secondary" %}}
run 포크 기능은 프라이빗 프리뷰 단계에 있습니다. 해당 기능 사용을 원하시면 support@wandb.com 으로 W&B 지원팀에 문의해 주세요.
{{% /alert %}}

[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ko" >}})에서 run 을 초기화할 때 `fork_from` 인수를 사용하면 기존 W&B run 을 "포크"할 수 있습니다. run 을 포크하면, W&B는 소스 run 의 `run ID`와 `step`을 기반으로 새로운 run 을 생성합니다.

run 을 포크하면, 실험의 특정 시점부터 원본 run 에 영향을 주지 않고 다양한 파라미터나 모델을 탐색할 수 있습니다.

{{% alert %}}
* run 을 포크하려면 [`wandb`](https://pypi.org/project/wandb/) SDK 버전이 0.16.5 이상이어야 합니다.
* run 을 포크할 때 step 값이 단조롭게 증가해야 합니다. [`define_metric()`]({{< relref path="/ref/python/sdk/classes/run#define_metric" lang="ko" >}})로 정의한 비단조적 step은 포크 지점으로 사용할 수 없습니다. 이는 run 히스토리와 시스템 메트릭의 필수 시간 순서를 방해하기 때문입니다.
{{% /alert %}}

## 포크한 run 시작하기

run 을 포크하려면, [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ko" >}})의 `fork_from` 인수를 활용하고, 포크할 소스 run 의 `run ID`와 `step`을 지정하세요:

```python
import wandb

# 나중에 포크할 run을 초기화합니다
original_run = wandb.init(project="your_project_name", entity="your_entity_name")
# ... 트레이닝 또는 로그 작업 수행 ...
original_run.finish()

# 특정 step에서 run 포크하기
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{original_run.id}?_step=200",
)
```

### 불변 run ID 사용하기

특정 run 을 항상 일관된 방식으로 참조하려면 불변 run ID 를 활용하세요. UI에서 불변 run ID 를 얻으려면 다음 단계를 따르세요:

1. **Overview 탭 엑세스:** 소스 run 페이지에서 [**Overview** 탭]({{< relref path="./#overview-tab" lang="ko" >}})으로 이동합니다.

2. **불변 run ID 복사:** **Overview** 탭 우측 상단의 `...` 메뉴(세 점 아이콘)를 클릭한 후 드롭다운에서 `Copy Immutable Run ID`를 선택하세요.

이 과정을 통해 run 을 포크할 때 사용할 수 있는 안정적인 run 참조 값을 얻을 수 있습니다.

## 포크한 run 이어서 사용하기
포크한 run 을 초기화한 뒤에는, 새로운 run 에 계속해서 로그를 남길 수 있습니다. 동일한 메트릭을 기록해 연속성을 유지하거나, 새로운 메트릭을 추가할 수도 있습니다.

예를 들어, 아래 코드 예시는 먼저 run 을 포크한 후, 트레이닝 step 200부터 포크한 run 에 메트릭을 로그하는 방법을 보여줍니다:

```python
import wandb
import math

# 첫 번째 run을 초기화하고 메트릭 기록하기
run1 = wandb.init("your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 첫 run의 특정 step에서 포크하고, 200번째 step부터 메트릭 기록 시작
run2 = wandb.init(
    "your_project_name", entity="your_entity_name", fork_from=f"{run1.id}?_step=200"
)

# 새 run에서 기록을 이어가기
# 처음 몇 step은 run1과 동일하게 메트릭을 기록
# 250번째 step 이후부터는 스파이크 패턴을 추가하여 기록
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i})  # run1의 기록을 스파이크 없이 그대로 이어갑니다
    else:
        # 250번째 step부터 스파이크 패턴 추가
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 미묘한 스파이크 패턴 적용
        run2.log({"metric": subtle_spike})
    # 모든 step에서 새로운 메트릭도 추가로 기록
    run2.log({"additional_metric": i * 1.1})
run2.finish()
```

{{% alert title="Rewind 및 포크의 호환성" %}}
포크 기능은 [`rewind`]({{< relref path="/guides/models/track/runs/rewind" lang="ko" >}})과 함께 사용하면 run 관리와 실험에 더 많은 유연성을 제공합니다.

run 을 포크하면, W&B는 run 의 특정 시점에서 분기를 만들어, 다른 파라미터나 모델을 실험할 수 있게 합니다.

반면, run 을 rewind 하면 W&B에서 run 히스토리 자체를 수정하거나 조정할 수 있습니다.
{{% /alert %}}