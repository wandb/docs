---
title: 실험별로 run 을 그룹화하기
description: 트레이닝 및 평가 run 을 더 큰 실험(Experiments) 단위로 그룹화하세요.
menu:
  default:
    identifier: ko-guides-models-track-runs-grouping
    parent: what-are-runs
---

개별 작업을 **group** 이름을 사용해 **wandb.init()** 에 전달함으로써 experiment로 그룹화할 수 있습니다.

## 유스 케이스

1. **분산 트레이닝:** experiment가 별도의 트레이닝 및 평가 스크립트 등 여러 조각으로 나뉘어 있을 때, 그룹화를 사용하여 이들을 하나의 큰 실험으로 볼 수 있습니다.
2. **다수의 프로세스:** 여러 개의 작은 프로세스를 하나의 experiment로 묶고 싶을 때 그룹화할 수 있습니다.
3. **K-폴드 교차 검증:** 서로 다른 랜덤 시드를 사용한 run을 그룹화하여 더 큰 experiment를 볼 수 있습니다. [k-fold 교차 검증 및 sweeps와 그룹화 예시](https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation)도 참고하세요.

그룹화는 여러 가지 방법으로 설정할 수 있습니다:

### 1. 스크립트에서 그룹 지정

`wandb.init()` 에 선택적으로 group과 `job_type` 을 전달하세요. 이렇게 하면 각 experiment마다 개별 run이 포함된 전용 group 페이지가 생깁니다. 예시:  
`wandb.init(group="experiment_1", job_type="eval")`

### 2. 환경 변수로 그룹 지정

`WANDB_RUN_GROUP` 환경 변수를 사용해서 run의 group을 지정할 수 있습니다. 자세한 내용은 [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}}) 문서를 참고하세요. **Group** 이름은 프로젝트 내에서 고유해야 하며, 해당 그룹의 모든 run이 공유해야 합니다. 모든 프로세스에서 사용할 고유 8자리 문자열을 생성하려면 `wandb.util.generate_id()` 를 사용할 수 있습니다. 예를 들어,  
`os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()`

### 3. UI에서 그룹 지정

run이 초기화된 후, Workspace나 **Runs** 페이지에서 새 그룹으로 이동할 수 있습니다.

1. W&B 프로젝트로 이동합니다.
1. 프로젝트 사이드바에서 **Workspace** 또는 **Runs** 탭을 선택합니다.
1. 이름을 바꿀 run을 검색하거나 스크롤하여 찾습니다.

    run 이름 위에 마우스를 올리고, 세로 점 3개(메뉴)를 클릭한 다음 **Move to another group**을 클릭하세요.
1. 새 그룹을 만들려면 **New group**을 클릭하고, 그룹 이름을 입력한 뒤 제출합니다.
1. 리스트에서 run의 새 그룹을 선택하고 **Move**를 클릭하세요.

### 4. UI에서 컬럼별 동적 그룹화

숨겨진 컬럼을 포함하여 어떤 컬럼이든 동적으로 그룹화할 수 있습니다. 예를 들어, `wandb.Run.config` 를 사용해 배치 크기나 러닝 레이트를 로그한 경우, 웹 앱에서 해당 하이퍼파라미터로 그룹화할 수 있습니다. **Group by** 기능은 [run group]({{< relref path="grouping.md" lang="ko" >}})과는 구별됩니다. run을 다른 run group으로 옮기려면 [UI에서 그룹 지정]({{< relref path="#set-a-group-in-the-ui" lang="ko" >}})을 참고하세요.

{{% alert %}}
run 리스트에서 **Group** 컬럼은 기본적으로 숨겨져 있습니다.
{{% /alert %}}

run을 하나 이상의 컬럼으로 그룹화하려면:

1. **Group**을 클릭하세요.
1. 하나 이상의 컬럼 이름을 클릭합니다.
1. 여러 컬럼을 선택했다면 드래그해서 그룹 순서를 조정할 수 있습니다.
1. 폼 외의 아무 곳이나 클릭해서 창을 닫으세요.

### run 표시 방식 커스터마이즈하기
프로젝트의 **Workspace** 또는 **Runs** 탭에서 run이 표시되는 방식을 커스터마이즈할 수 있습니다. 두 탭은 동일한 표시 설정을 사용합니다.

표시할 컬럼을 선택하려면:
1. run 리스트 상단에서 **Columns**를 클릭합니다.
1. 숨겨진 컬럼의 이름을 클릭하면 표시되고, 보이는 컬럼 이름을 클릭하면 숨겨집니다.
    
    컬럼 이름으로 퍼지 검색, 정확 일치, 정규표현식 등으로 검색할 수 있습니다. 컬럼을 드래그해서 순서를 바꿀 수 있습니다.
1. **Done**을 클릭해 컬럼 브라우저를 닫으세요.

표시된 어떤 컬럼으로든 run 목록을 정렬하려면:

1. 컬럼 이름 위에 마우스를 올리고, 액션 메뉴(`...`)를 클릭합니다.
1. **Sort ascending** 또는 **Sort descending**을 클릭합니다.

고정 컬럼은 오른쪽에 표시됩니다. 컬럼을 고정하거나 해제하려면:
1. 컬럼 이름 위에 마우스를 올리고, 액션 메뉴(`...`)를 클릭합니다.
1. **Pin column** 또는 **Unpin column**을 클릭하세요.

기본적으로 긴 run 이름은 가독성을 위해 중간이 생략됩니다. run 이름의 생략 위치를 커스터마이즈하려면:

1. run 리스트 상단의 액션 메뉴(`...`)를 클릭하세요.
1. **Run name cropping**에서 시작, 중간, 끝 중 어디를 생략할지 정할 수 있습니다.

## 그룹화와 분산 트레이닝

만약 `wandb.init()` 에서 그룹을 지정하면, UI에서 run이 자동으로 그룹화되어 표시됩니다. 표 상단의 **Group** 버튼을 클릭해서 그룹화 표시를 껐다 켰다 할 수 있습니다. [샘플 코드](https://wandb.me/grouping)로 만든 [예시 프로젝트](https://wandb.ai/carey/group-demo?workspace=user-carey)에서 그룹화를 지정한 케이스를 볼 수 있습니다. 사이드바의 각 "Group" 행을 클릭하면 해당 experiment의 전용 그룹 페이지로 이동할 수 있습니다.

{{< img src="/images/track/distributed_training_wgrouping_1.png" alt="Grouped runs view" >}}

위 프로젝트 페이지에서 왼쪽 사이드바의 **Group**을 클릭하면, [이런 전용 페이지](https://wandb.ai/carey/group-demo/groups/exp_5?workspace=user-carey)로 이동합니다:

{{< img src="/images/track/distributed_training_wgrouping_2.png" alt="Group details page" >}}

## UI에서 동적으로 그룹화하기

run을 어떤 컬럼으로든 그룹화할 수 있습니다. 예를 들어, 하이퍼파라미터별 그룹화가 가능합니다. 아래는 그 예시입니다:

* **사이드바:** run이 에포크 수에 따라 그룹화됨
* **그래프:** 각 선은 그룹의 평균을 나타내고, 음영은 분산을 뜻합니다. 이 행동은 그래프 설정에서 변경할 수 있습니다.

{{< img src="/images/track/demo_grouping.png" alt="Dynamic grouping by epochs" >}}

## 그룹화 끄기

언제든 그룹화 버튼을 클릭해 그룹 필드를 모두 비우면, 표와 그래프가 그룹화되지 않은 상태로 돌아갑니다.

{{< img src="/images/track/demo_no_grouping.png" alt="Ungrouped runs table" >}}

## 그룹화 그래프 설정

그래프 오른쪽 위의 편집 버튼을 클릭해 **Advanced** 탭에서 선과 음영 표시를 바꿀 수 있습니다. 각 그룹의 선을 평균, 최소, 최대값 등으로 설정할 수 있습니다. 음영은 끄거나, 최소-최대, 표준 편차, 표준 오차 등으로 표시할 수 있습니다.

{{< img src="/images/track/demo_grouping_options_for_line_plots.gif" alt="Line plot grouping options" >}}