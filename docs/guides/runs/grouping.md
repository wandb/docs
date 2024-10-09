---
title: Group runs into experiments
description: 트레이닝 및 평가 run을 더 큰 실험으로 그룹화하기
displayed_sidebar: default
---

개별 작업을 고유한 **group** 이름을 **wandb.init()**에 전달하여 Experiments로 그룹화하세요.

## 유스 케이스

1. **분산 트레이닝:** 실험이 더 큰 전체의 일부로서 별도의 트레이닝 및 평가 스크립트로 분할된 경우 그룹화를 사용하세요.
2. **다중 프로세스:** 여러 개의 작은 프로세스를 하나의 실험으로 그룹화하세요.
3. **K-폴드 교차 검증:** 다른 랜덤 시드를 가진 runs를 하나의 큰 Experiment로 그룹화하세요. 여기 [예시](https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation)가 있습니다.

그룹 설정 방법은 세 가지가 있습니다:

### 1. 스크립트 내에서 그룹 설정하기

group과 job_type을 wandb.init()에 선택적으로 전달하십시오. 이렇게 하면 개별 runs를 포함한 각 experiment를 위한 전용 그룹 페이지를 얻을 수 있습니다. 예를 들어: `wandb.init(group="experiment_1", job_type="eval")`

### 2. 그룹 환경 변수 설정하기

환경 변수로서 `WANDB_RUN_GROUP`을 사용하여 runs의 그룹을 지정하세요. 이에 대한 추가 정보는 [**환경 변수**](../track/environment-variables.md)**를** 참조하세요. **Group**은 프로젝트 내에서 고유해야 하며 그룹 내 모든 runs에 의해 공유되어야 합니다. `wandb.util.generate_id()`를 사용하여 모든 프로세스에 사용할 수 있는 고유한 8자 문자열을 생성할 수 있습니다. 예를 들어, `os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()`

### 3. UI에서 그룹화 전환하기

어떤 설정 열이라도 동적으로 그룹화할 수 있습니다. 예를 들어, `wandb.config`를 사용하여 배치 크기나 학습률을 로그할 경우, 웹 앱에서 이러한 하이퍼파라미터에 따라 동적으로 그룹화할 수 있습니다.

## 그룹화를 활용한 분산 트레이닝

`wandb.init()`에서 그룹화를 설정했다고 가정하면, 우리는 UI에서 기본적으로 runs를 그룹화할 것입니다. 테이블 상단의 **Group** 버튼을 클릭하여 이를 켜고 끌 수 있습니다. 우리가 그룹화를 설정한 [샘플 코드](http://wandb.me/grouping)로 생성된 [예시 프로젝트](https://wandb.ai/carey/group-demo?workspace=user-carey)를 참조하십시오. 사이드바에서 각 "Group" 행을 클릭하여 해당 experiment에 대한 전용 그룹 페이지로 이동할 수 있습니다.

![](/images/track/distributed_training_wgrouping_1.png)

위의 프로젝트 페이지에서 좌측 사이드바에 있는 **Group**을 클릭하여 [이와 같은](https://wandb.ai/carey/group-demo/groups/exp_5?workspace=user-carey) 전용 페이지로 이동할 수 있습니다.

![](/images/track/distributed_training_wgrouping_2.png)

## UI에서 동적으로 그룹화하기

어떤 열이라도 그룹화할 수 있으며, 예를 들어 하이퍼파라미터에 따라 그룹화할 수 있습니다. 그 예시는 다음과 같습니다:

* **사이드바**: Runs는 에포크 수에 따라 그룹화됩니다.
* **그래프**: 각 선은 그룹의 평균을 나타내며, 음영은 분산을 나타냅니다. 이 행동은 그래프 설정에서 변경할 수 있습니다.

![](/images/track/demo_grouping.png)

## 그룹화 끄기

그룹화 버튼을 클릭하고 언제든지 그룹 필드를 지워 테이블과 그래프를 그룹화되지 않은 상태로 되돌리세요.

![](/images/track/demo_no_grouping.png)

## 그룹화된 그래프 설정

그래프의 오른쪽 상단 모서리에 있는 편집 버튼을 클릭한 다음 **고급** 탭을 선택하여 선과 음영을 변경하십시오. 각 그룹의 선에 대해 평균, 최소값 또는 최대값을 선택할 수 있습니다. 음영의 경우, 음영을 끄거나 최소 및 최대, 표준 편차, 표준 오차를 표시할 수 있습니다.

![](/images/track/demo_grouping_options_for_line_plots.gif)

## 자주 묻는 질문

### 태그로 runs를 그룹화할 수 있나요?

Run은 여러 태그를 가질 수 있으므로 이 필드로 그룹화를 지원하지 않습니다. 우리의 권장 사항은 이러한 runs의 [`config`](../track/config.md) 오브젝트에 값을 추가한 다음 이 구성 값으로 그룹화하는 것입니다. 이것은 [우리의 API](../track/config#set-the-configuration-after-your-run-has-finished)를 사용하여 수행할 수 있습니다.