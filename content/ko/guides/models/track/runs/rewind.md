---
title: run 되돌리기
description: 되감기
menu:
  default:
    identifier: ko-guides-models-track-runs-rewind
    parent: what-are-runs
---

# Run 되감기
{{% alert color="secondary" %}}
Run 되감기 옵션은 현재 프라이빗 프리뷰 단계에 있습니다. 이 기능을 사용하고 싶으시다면 support@wandb.com 으로 W&B Support에 문의해 주세요.

W&B는 현재 아래 기능을 지원하지 않습니다:
* **로그 되감기**: 새로운 run 구간에서는 로그가 초기화됩니다.
* **시스템 메트릭 되감기**: 되감기 지점 이후의 새로운 시스템 메트릭만 W&B가 기록합니다.
* **Artifact 연관**: W&B는 artifact를 생성한 소스 run과 연관시킵니다.
{{% /alert %}}

{{% alert %}}
* Run 되감기를 위해서는 [W&B Python SDK](https://pypi.org/project/wandb/) 버전이 `0.17.1` 이상이어야 합니다.
* 단조 증가하는 step을 사용해야 합니다. [`define_metric()`]({{< relref path="/ref/python/sdk/classes/run#define_metric" lang="ko" >}})으로 정의된 비단조 step에서는 동작하지 않으며, 이는 run 히스토리와 시스템 메트릭의 필수적인 시간 순서를 깨뜨리기 때문입니다.
{{% /alert %}}

Run의 히스토리를 수정하거나 보정할 때 원본 데이터를 잃지 않고 run을 되감아 보세요. 또한, 되감기를 통해 해당 시점부터 새로운 데이터를 기록할 수 있습니다. W&B는 되감기한 run의 요약 메트릭을 새로 기록된 히스토리를 바탕으로 다시 계산합니다. 주요 동작은 다음과 같습니다.
- **히스토리 절단**: 되감기 지점까지 히스토리를 절단하여 이후 데이터 기록이 가능합니다.
- **요약 메트릭**: 새로 로그된 히스토리를 바탕으로 다시 계산됩니다.
- **설정값 보존**: 원본 설정이 유지되고, 새로운 설정도 병합할 수 있습니다.

run을 되감기 하면 W&B는 해당 run의 상태를 지정한 step까지 초기화하지만, 원본 데이터는 보존하며 run ID의 일관성도 유지합니다. 주요 사항은 다음과 같습니다.

- **Run 보관**: W&B는 원본 run들을 아카이브합니다. Run들은 [Run Overview]({{< relref path="./#overview-tab" lang="ko" >}}) 탭에서 엑세스할 수 있습니다.
- **Artifact 연관**: Artifact는 그것을 생성한 run과 연관됩니다.
- **불변 run ID**: 정확한 상태에서 포크가 가능하도록 불변의 run ID가 도입됩니다.
- **불변 run ID 복사**: run 관리 향상을 위해 불변 run ID를 복사할 수 있는 버튼이 추가됩니다.

{{% alert title="Rewind와 Forking의 호환성" %}}
Forking은 rewind 기능을 보완합니다.

Fork를 사용할 경우, 특정 시점의 run에서 새로운 브랜치를 만들어 여러 파라미터나 모델을 실험할 수 있습니다.

Rewind는 run 히스토리 자체를 수정하거나 보정하도록 해줍니다.
{{% /alert %}}



## Run 되감기 사용법

`wandb.init()`의 `resume_from` 옵션을 활용해 run의 히스토리를 지정한 step까지 “되감기”할 수 있습니다. 되감기 하고자 하는 run의 이름과 시작할 step을 지정해 주세요:

```python
import wandb
import math

# 첫 번째 run을 초기화하고 일부 메트릭을 기록합니다.
# your_project_name 과 your_entity_name을 실제 값으로 바꿔주세요!
run1 = wandb.init(project="your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# 첫 번째 run을 특정 step에서 되감기하고, step 200부터 메트릭을 기록합니다.
run2 = wandb.init(project="your_project_name", entity="your_entity_name", resume_from=f"{run1.id}?_step=200")

# 새로운 run에서 계속해서 로그를 남깁니다.
# 처음 몇 step 동안은 run1의 메트릭 그대로 기록
# step 250 이후에는 스파이크 패턴을 기록
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i, "step": i})  # run1에서 이어서 스파이크 없이 기록
    else:
        # step 250 부터 스파이키한 패턴을 적용
        subtle_spike = i + (2 * math.sin(i / 3.0))  # 은은한 스파이크 패턴 적용
        run2.log({"metric": subtle_spike, "step": i})
    # 모든 step에서 추가 메트릭 기록
    run2.log({"additional_metric": i * 1.1, "step": i})
run2.finish()
```

## 아카이브된 run 확인하기

run을 되감기한 후에는, W&B App UI에서 아카이브된 run을 조회할 수 있습니다. 아래 순서로 진행하세요:

1. **Overview 탭 엑세스:** run 페이지에서 [**Overview** 탭]({{< relref path="./#overview-tab" lang="ko" >}})으로 이동합니다. 이 탭에서 run의 상세 내용과 히스토리를 확인할 수 있습니다.
2. **Forked From 필드 찾기:** **Overview** 탭에서 `Forked From` 필드를 찾으세요. 이 필드에는 resume 관련 히스토리가 남아 있습니다. **Forked From** 필드는 소스 run으로 연결되는 링크가 포함되어 있어, 오리지널 run까지 추적 및 전체 rewind 히스토리를 이해할 수 있습니다.

`Forked From` 필드를 활용하면 아카이브된 resume의 트리를 간편하게 탐색하며, 각 rewind의 순서와 출처를 파악할 수 있습니다.

## 되감기된 run에서 포크하기

되감기한 run에서 포크하려면, `wandb.init()`의 [`fork_from`]({{< relref path="/guides/models/track/runs/forking" lang="ko" >}}) 인수를 사용하고, 소스 run의 ID와 시작할 step을 지정하세요:

```python 
import wandb

# 특정 step에서 run을 포크합니다.
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{rewind_run.id}?_step=500",
)

# 새로운 run에서 계속 로그를 남깁니다.
for i in range(500, 1000):
    forked_run.log({"metric": i*3})
forked_run.finish()
```