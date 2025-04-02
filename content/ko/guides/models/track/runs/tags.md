---
title: Add labels to runs with tags
menu:
  default:
    identifier: ko-guides-models-track-runs-tags
    parent: what-are-runs
---

로그된 메트릭 또는 아티팩트 데이터에서 명확하지 않을 수 있는 특정 기능으로 Runs에 레이블을 지정하려면 태그를 추가하세요.

예를 들어, Run의 모델이 `in_production`임을 나타내거나, 해당 Run이 `preemptible`인지, 이 Run이 `baseline`을 나타내는지 등을 나타내는 태그를 Run에 추가할 수 있습니다.

## 하나 이상의 Runs에 태그 추가

프로그래밍 방식으로 또는 대화형으로 Runs에 태그를 추가합니다.

사용 사례에 따라 필요에 가장 적합한 아래 탭을 선택하세요.

{{< tabpane text=true >}}
    {{% tab header="W&B Python SDK" %}}
Run이 생성될 때 태그를 추가할 수 있습니다.

```python
import wandb

run = wandb.init(
  entity="entity",
  project="<project-name>",
  tags=["tag1", "tag2"]
)
```

Run을 초기화한 후 태그를 업데이트할 수도 있습니다. 예를 들어, 다음 코드 조각은 특정 메트릭이 미리 정의된 임계값을 넘는 경우 태그를 업데이트하는 방법을 보여줍니다.

```python
import wandb

run = wandb.init(
  entity="entity", 
  project="capsules", 
  tags=["debug"]
  )

# 모델을 훈련하는 파이썬 로직

if current_loss < threshold:
    run.tags = run.tags + ("release_candidate",)
```    
    {{% /tab %}}
    {{% tab header="Public API" %}}
Run을 생성한 후에는 [Public API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ko" >}})를 사용하여 태그를 업데이트할 수 있습니다. 예를 들면 다음과 같습니다.

```python
run = wandb.Api().run("{entity}/{project}/{run-id}")
run.tags.append("tag1")  # 여기에서 Run 데이터를 기반으로 태그를 선택할 수 있습니다.
run.update()
```    
    {{% /tab %}}
    {{% tab header="Project page" %}}
이 방법은 동일한 태그를 사용하여 많은 수의 Runs에 태그를 지정하는 데 가장 적합합니다.

1. 프로젝트 워크스페이스로 이동합니다.
2. 프로젝트 사이드바에서 **Runs**를 선택합니다.
3. 테이블에서 하나 이상의 Runs를 선택합니다.
4. 하나 이상의 Runs를 선택했으면 테이블 위의 **Tag** 버튼을 선택합니다.
5. 추가할 태그를 입력하고 **Create new tag** 확인란을 선택하여 태그를 추가합니다.
    {{% /tab %}}
    {{% tab header="Run page" %}}
이 방법은 단일 Run에 수동으로 태그를 적용하는 데 가장 적합합니다.

1. 프로젝트 워크스페이스로 이동합니다.
2. 프로젝트 워크스페이스 내 Runs 목록에서 Run을 선택합니다.
3. 프로젝트 사이드바에서 **Overview**를 선택합니다.
4. **Tags** 옆에 있는 회색 더하기 아이콘(**+**) 버튼을 선택합니다.
5. 추가할 태그를 입력하고 텍스트 상자 아래의 **Add**를 선택하여 새 태그를 추가합니다.
    {{% /tab %}}
{{< /tabpane >}}

## 하나 이상의 Runs에서 태그 제거

태그는 W&B App UI를 사용하여 Runs에서 제거할 수도 있습니다.

{{< tabpane text=true >}}
{{% tab header="Project page"%}}
이 방법은 많은 수의 Runs에서 태그를 제거하는 데 가장 적합합니다.

1. 프로젝트의 Run 사이드바에서 오른쪽 상단의 테이블 아이콘을 선택합니다. 이렇게 하면 사이드바가 전체 Runs 테이블로 확장됩니다.
2. 테이블에서 Run 위로 마우스를 가져가면 왼쪽에 확인란이 표시되거나 헤더 행에서 모든 Runs를 선택하는 확인란을 찾습니다.
3. 확인란을 선택하여 대량 작업을 활성화합니다.
4. 태그를 제거할 Runs를 선택합니다.
5. Runs 행 위의 **Tag** 버튼을 선택합니다.
6. 태그 옆에 있는 확인란을 선택하여 Run에서 제거합니다.

{{% /tab %}}
{{% tab header="Run page"%}}

1. Run 페이지의 왼쪽 사이드바에서 맨 위 **Overview** 탭을 선택합니다. Run의 태그가 여기에 표시됩니다.
2. 태그 위로 마우스를 가져간 다음 "x"를 선택하여 Run에서 제거합니다.

{{% /tab %}}
{{< /tabpane >}}
