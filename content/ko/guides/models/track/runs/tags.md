---
title: 태그로 run 에 라벨 추가하기
menu:
  default:
    identifier: ko-guides-models-track-runs-tags
    parent: what-are-runs
---

특정 특징을 가진 run 에 태그를 추가하여, 로그된 메트릭이나 아티팩트 데이터만으로는 알 수 없는 정보를 라벨링할 수 있습니다.

예를 들어, run 의 모델이 `in_production` 상태임을 표시하거나, 해당 run 이 `preemptible` 이거나, 이 run 이 `baseline` 을 대표함을 나타내는 태그 등을 추가할 수 있습니다.

## 하나 이상의 run 에 태그 추가하기

프로그램적으로 또는 인터랙티브하게 run 에 태그를 추가할 수 있습니다.

유스 케이스에 따라 아래 탭 중 가장 잘 맞는 방법을 선택해 주세요.

{{< tabpane text=true >}}
    {{% tab header="W&B Python SDK" %}}
run 이 생성될 때 태그를 추가할 수 있습니다.

```python
import wandb

run = wandb.init(
  entity="entity",
  project="<project-name>",
  tags=["tag1", "tag2"]
)
```

run 을 초기화한 후에도 태그를 업데이트할 수 있습니다. 예를 들어, 다음 코드조각은 특정 메트릭이 미리 정의된 임계값을 넘었을 때 태그를 업데이트하는 방법을 보여줍니다:

```python
import wandb

run = wandb.init(
  entity="entity", 
  project="capsules", 
  tags=["debug"]
  )

# 모델을 학습시키는 파이썬 로직

if current_loss < threshold:
    run.tags = run.tags + ("release_candidate",)
```    
    {{% /tab %}}
    {{% tab header="Public API" %}}
run 을 생성한 후에는 [Public API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ko" >}})를 활용해 태그를 업데이트할 수 있습니다. 예시:

```python
run = wandb.Api().run("{entity}/{project}/{run-id}")
run.tags.append("tag1")  # run 데이터에 따라 태그를 선택할 수 있습니다
run.update()
```    
    {{% /tab %}}
    {{% tab header="Project page" %}}
이 방법은 동일한 태그를 다수의 run 에 한 번에 적용하기에 가장 적합합니다.

1. 프로젝트 워크스페이스로 이동합니다.
2. 프로젝트 사이드바에서 **Runs** 를 선택합니다.
3. 테이블에서 하나 이상의 run 을 선택합니다.
4. 하나 이상의 run 을 선택했다면, 테이블 상단의 **Tag** 버튼을 선택합니다.
5. 추가할 태그를 입력하고 **Create new tag** 체크박스를 선택하여 새로운 태그를 추가합니다.    
    {{% /tab %}}
    {{% tab header="Run page" %}}
이 방법은 단일 run 에 태그를 직접 추가할 때 가장 적합합니다.

1. 프로젝트 워크스페이스로 이동합니다.
2. 프로젝트의 워크스페이스에 있는 run 목록에서 run 하나를 선택합니다.
1. 프로젝트 사이드바에서 **Overview** 를 선택합니다.
2. **Tags** 옆의 회색 플러스 아이콘 (**+**) 버튼을 선택합니다.
3. 추가할 태그를 입력한 후, 텍스트 박스 아래의 **Add** 를 선택해 새로운 태그를 추가합니다.    
    {{% /tab %}}
{{< /tabpane >}}



## 하나 이상의 run 에서 태그 제거하기

태그는 W&B App UI 를 통해 run 에서 제거할 수도 있습니다.

{{< tabpane text=true >}}
{{% tab header="Project page"%}}
이 방법은 다수의 run 에서 태그를 한 번에 제거할 때 가장 적합합니다.

1. 프로젝트의 Run 사이드바에서 오른쪽 상단의 테이블 아이콘을 클릭합니다. 사이드바가 확장되어 전체 run 테이블이 표시됩니다.
2. 테이블에서 run 위에 마우스를 올려두면 왼쪽에 체크박스가 나타나거나, 헤더 행에서 전체 선택용 체크박스를 볼 수 있습니다.
3. 체크박스를 선택해 여러 run 에 대한 일괄 작업을 활성화합니다.
4. 태그를 제거할 run 들을 선택합니다.
5. run 들 위에 있는 **Tag** 버튼을 선택합니다.
6. run 에서 제거하려는 태그 옆의 체크박스를 선택 해제하면 태그가 삭제됩니다.

{{% /tab %}}
{{% tab header="Run page"%}}

1. Run 페이지 왼쪽 사이드바에서 상단의 **Overview** 탭을 클릭합니다. run 에 달린 태그가 이곳에 보입니다.
2. 삭제할 태그 위에 마우스를 올리면 "x" 가 나타나는데, 클릭하면 run 에서 해당 태그가 제거됩니다.

{{% /tab %}}
{{< /tabpane >}}