---
description: Group training and evaluation runs into larger experiments
displayed_sidebar: default
---

# 실행 그룹화

<head>
  <title>W&B 실행 그룹화</title>
</head>


**wandb.init()**에 고유한 **그룹** 이름을 전달하여 개별 작업을 실험으로 그룹화합니다.

## 사용 사례

1. **분산 학습:** 실험이 별도의 학습 및 평가 스크립트로 나뉘어져 있고, 이를 하나의 큰 전체로 보아야 할 경우 그룹화를 사용합니다.
2. **다중 프로세스:** 여러 개의 작은 프로세스를 하나의 실험으로 그룹화합니다.
3. **K-폴드 교차 검증:** 다른 랜덤 시드를 가진 실행을 그룹화하여 더 큰 실험을 확인합니다. 여기 [스윕과 그룹화를 사용한 k-폴드 교차 검증의 예](https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation)가 있습니다.

그룹화를 설정하는 세 가지 방법은 다음과 같습니다:

### 1. 스크립트에서 그룹 설정

wandb.init()에 선택적 그룹과 job_type을 전달합니다. 이렇게 하면 각 실험마다 전용 그룹 페이지가 제공되며, 개별 실행이 포함됩니다. 예를 들어: `wandb.init(group="experiment_1", job_type="eval")`

### 2. 그룹 환경 변수 설정

`WANDB_RUN_GROUP`을 사용하여 실행에 대한 그룹을 환경 변수로 지정합니다. 자세한 내용은 [**환경 변수**](../track/environment-variables.md) 문서를 확인하세요. **그룹**은 프로젝트 내에서 고유해야 하며 그룹의 모든 실행에서 공유되어야 합니다. 예를 들어, `os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()`와 같이 `wandb.util.generate_id()`를 사용하여 모든 프로세스에서 사용할 고유한 8자리 문자열을 생성할 수 있습니다.

### 3. UI에서 그룹화 토글

배치 크기나 학습률과 같은 `wandb.config`을 사용하여 로그를 기록한 경우, 웹 앱에서 동적으로 이러한 하이퍼파라미터별로 그룹화할 수 있습니다.

## 그룹화를 통한 분산 학습

`wandb.init()`에서 그룹화를 설정하면 기본적으로 UI에서 실행을 그룹화합니다. 테이블 상단의 **그룹** 버튼을 클릭하여 이를 켜고 끌 수 있습니다. 그룹화를 설정한 [예시 프로젝트](https://wandb.ai/carey/group-demo?workspace=user-carey)가 있으며, 여기에서 각 "그룹" 행을 클릭하면 해당 실험의 전용 그룹 페이지로 이동할 수 있습니다.

![](/images/track/distributed_training_wgrouping_1.png)

위의 프로젝트 페이지에서 왼쪽 사이드바의 **그룹**을 클릭하면 [이 페이지](https://wandb.ai/carey/group-demo/groups/exp\_5?workspace=user-carey)와 같은 전용 페이지로 이동할 수 있습니다:

![](/images/track/distributed_training_wgrouping_2.png)

## UI에서 동적으로 그룹화

예를 들어 하이퍼파라미터별로 실행을 그룹화할 수 있습니다. 그룹화가 어떻게 보이는지 예시는 다음과 같습니다:

* **사이드바**: 실행은 에포크의 수에 따라 그룹화됩니다.
* **그래프**: 각 선은 그룹의 평균을 나타내며, 음영은 변동성을 나타냅니다. 이 동작은 그래프 설정에서 변경할 수 있습니다.

![](/images/track/demo_grouping.png)

## 그룹화 해제

그룹화 버튼을 클릭하고 그룹 필드를 언제든지 지워서 테이블과 그래프를 그룹화되지 않은 상태로 돌릴 수 있습니다.

![](/images/track/demo_no_grouping.png)

## 그룹화 그래프 설정

그래프의 오른쪽 상단에 있는 편집 버튼을 클릭하고 **고급** 탭을 선택하여 선과 음영을 변경합니다. 각 그룹에서 선에 대해 평균, 최소 또는 최대 값을 선택할 수 있습니다. 음영에 대해서는 음영을 끄고, 최소 및 최대, 표준 편차, 표준 오차를 표시할 수 있습니다.

![](/images/track/demo_grouping_options_for_line_plots.gif)

## 자주 묻는 질문

### 태그별로 실행을 그룹화할 수 있나요?

실행에 여러 태그가 있을 수 있으므로 이 필드별로 그룹화를 지원하지 않습니다. 저희의 권장 사항은 이러한 실행의 [`config`](../track/config.md) 객체에 값을 추가한 다음 이 config 값으로 그룹화하는 것입니다. [저희 API](../track/config.md#update-config-files)를 사용하여 이를 수행할 수 있습니다.