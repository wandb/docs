---
description: Group training and evaluation runs into larger experiments
displayed_sidebar: default
---

# 실행 그룹화

<head>
  <title>W&B 실행 그룹화</title>
</head>


독특한 **group** 이름을 **wandb.init()**에 전달하여 개별 작업을 실험으로 그룹화합니다.

## 유스 케이스

1. **분산 트레이닝:** 실험이 별도의 트레이닝 및 평가 스크립트로 나뉘어져 있고, 이를 더 큰 전체의 일부로 보아야 하는 경우 그룹화를 사용합니다.
2. **여러 프로세스:** 여러 작은 프로세스들을 하나의 실험으로 그룹화합니다.
3. **K-fold 교차 검증:** 다른 랜덤 시드를 가진 실행들을 그룹화하여 더 큰 실험을 볼 수 있습니다. 여기 스윕과 그룹화를 사용한 k-fold 교차 검증의 [예시](https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation)가 있습니다.

그룹화를 설정하는 세 가지 방법이 있습니다:

### 1. 스크립트에서 그룹 설정

wandb.init()에 선택적 그룹과 job_type을 전달합니다. 이를 통해 각 실험에 대한 전용 그룹 페이지가 제공되며, 개별 실행이 포함됩니다. 예를 들어:`wandb.init(group="experiment_1", job_type="eval")`

### 2. 그룹 환경 변수 설정

`WANDB_RUN_GROUP`을 사용하여 환경 변수로 실행의 그룹을 지정합니다. 이에 대한 자세한 내용은 [**환경 변수**](../track/environment-variables.md)**. 그룹**은 프로젝트 내에서 고유해야 하며 그룹 내의 모든 실행이 공유해야 합니다. `wandb.util.generate_id()`를 사용하여 모든 프로세스에 사용할 고유한 8자리 문자열을 생성할 수 있습니다. 예를 들어, `os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()`

### 3. UI에서 그룹화 전환

배치 크기나 학습률과 같은 하이퍼파라미터를 로그하기 위해 `wandb.config`를 사용하는 경우, 웹 앱에서 동적으로 이러한 하이퍼파라미터별로 그룹화할 수 있습니다.

## 그룹화를 사용한 분산 트레이닝

`wandb.init()`에서 그룹화를 설정하면, UI에서 기본적으로 실행을 그룹화합니다. 테이블 상단의 **Group** 버튼을 클릭하여 이를 켜고 끌 수 있습니다. 그룹화를 설정한 [샘플 코드](http://wandb.me/grouping)에서 생성된 [예시 프로젝트](https://wandb.ai/carey/group-demo?workspace=user-carey)가 있습니다. 사이드바에서 각 "Group" 행을 클릭하여 해당 실험의 전용 그룹 페이지로 이동할 수 있습니다.

![](/images/track/distributed_training_wgrouping_1.png)

위 프로젝트 페이지에서, 왼쪽 사이드바의 **Group**을 클릭하면 [이 페이지](https://wandb.ai/carey/group-demo/groups/exp\_5?workspace=user-carey)와 같은 전용 페이지로 이동할 수 있습니다:

![](/images/track/distributed_training_wgrouping_2.png)

## UI에서 동적으로 그룹화

하이퍼파라미터와 같은 모든 열별로 실행을 그룹화할 수 있습니다. 이것이 어떤 모습인지 예시입니다:

* **사이드바**: 에포크 수에 따라 실행이 그룹화됩니다.
* **그래프**: 각 선은 그룹의 평균을 나타내며, 음영은 분산을 나타냅니다. 이 행동은 그래프 설정에서 변경할 수 있습니다.

![](/images/track/demo_grouping.png)

## 그룹화 해제

그룹화 버튼을 클릭하고 그룹 필드를 언제든지 지워 테이블과 그래프를 그룹화되지 않은 상태로 되돌릴 수 있습니다.

![](/images/track/demo_no_grouping.png)

## 그룹화 그래프 설정

그래프의 오른쪽 상단에 있는 편집 버튼을 클릭하고 **고급** 탭을 선택하여 선과 음영을 변경합니다. 각 그룹에서 선의 값을 평균, 최소 또는 최대 값으로 선택할 수 있습니다. 음영의 경우, 음영을 끄고, 최소 및 최대, 표준 편차, 표준 오차를 표시할 수 있습니다.

![](/images/track/demo_grouping_options_for_line_plots.gif)

## 자주 묻는 질문

### 태그별로 실행을 그룹화할 수 있나요?

실행이 여러 태그를 가질 수 있기 때문에 이 필드로 그룹화를 지원하지 않습니다. 우리의 추천은 이러한 실행의 [`config`](../track/config.md) 오브젝트에 값을 추가하고 이 config 값으로 그룹화하는 것입니다. 이는 [우리의 API](../track/config.md#update-config-files)로 할 수 있습니다.