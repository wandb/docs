---
title: Group runs into experiments
description: 트레이닝 과 평가 runs 을 더 큰 Experiments 로 그룹화합니다
menu:
  default:
    identifier: ko-guides-models-track-runs-grouping
    parent: what-are-runs
---

**wandb.init()** 에 고유한 **group** 이름을 전달하여 개별 작업을 Experiments 로 그룹화합니다.

## 유스 케이스

1. **분산 트레이닝:** 더 큰 전체의 일부로 간주해야 하는 별도의 트레이닝 및 평가 스크립트로 Experiments 가 분할된 경우 그룹화를 사용합니다.
2. **다중 프로세스**: 여러 개의 작은 프로세스를 함께 묶어 experiment 로 만듭니다.
3. **K-fold 교차 검증**: 다른 랜덤 시드를 사용하여 Runs 를 함께 그룹화하여 더 큰 experiment 를 확인합니다. 다음은 스윕 및 그룹화를 사용한 k-fold 교차 검증 [예시](https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation) 입니다.

그룹 설정을 하는 세 가지 방법이 있습니다.

### 1. 스크립트에서 그룹 설정

wandb.init() 에 선택적 group 및 job_type 을 전달합니다. 그러면 각 experiment 에 대한 전용 그룹 페이지가 제공되며, 여기에는 개별 Runs 가 포함됩니다. 예: `wandb.init(group="experiment_1", job_type="eval")`

### 2. 그룹 환경 변수 설정

`WANDB_RUN_GROUP` 를 사용하여 Runs 에 대한 그룹을 환경 변수로 지정합니다. 자세한 내용은 [**환경 변수**]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}}) 에 대한 문서를 확인하세요. **Group** 은 프로젝트 내에서 고유해야 하며 그룹의 모든 Runs 에서 공유해야 합니다. `wandb.util.generate_id()` 를 사용하여 모든 프로세스에서 사용할 고유한 8자 문자열을 생성할 수 있습니다 (예: `os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()`).

### 3. UI 에서 그룹 전환

구성 열을 기준으로 동적으로 그룹화할 수 있습니다. 예를 들어 `wandb.config` 를 사용하여 배치 크기 또는 학습 속도를 기록한 다음 웹 앱에서 해당 하이퍼파라미터별로 동적으로 그룹화할 수 있습니다.

## 그룹화를 사용한 분산 트레이닝

`wandb.init()` 에서 그룹 설정을 했다고 가정하면 UI 에서 기본적으로 Runs 가 그룹화됩니다. 테이블 상단의 **Group** 버튼을 클릭하여 켜거나 끌 수 있습니다. 그룹 설정을 한 [샘플 코드](http://wandb.me/grouping) 에서 생성된 [예제 프로젝트](https://wandb.ai/carey/group-demo?workspace=user-carey) 입니다. 사이드바에서 각 "Group" 행을 클릭하여 해당 experiment 에 대한 전용 그룹 페이지로 이동할 수 있습니다.

{{< img src="/images/track/distributed_training_wgrouping_1.png" alt="" >}}

위의 프로젝트 페이지에서 왼쪽 사이드바의 **Group** 을 클릭하여 [이 페이지](https://wandb.ai/carey/group-demo/groups/exp_5?workspace=user-carey) 와 같은 전용 페이지로 이동할 수 있습니다.

{{< img src="/images/track/distributed_training_wgrouping_2.png" alt="" >}}

## UI 에서 동적으로 그룹화

예를 들어 하이퍼파라미터별로 열을 기준으로 Runs 를 그룹화할 수 있습니다. 다음과 같은 모양의 예가 있습니다.

* **사이드바**: Runs 가 epoch 수로 그룹화됩니다.
* **그래프**: 각 선은 그룹의 평균을 나타내고 음영은 분산을 나타냅니다. 이 동작은 그래프 설정에서 변경할 수 있습니다.

{{< img src="/images/track/demo_grouping.png" alt="" >}}

## 그룹 해제

언제든지 그룹화 버튼을 클릭하고 그룹 필드를 지우면 테이블과 그래프가 그룹 해제된 상태로 돌아갑니다.

{{< img src="/images/track/demo_no_grouping.png" alt="" >}}

## 그래프 설정 그룹화

그래프의 오른쪽 상단 모서리에 있는 편집 버튼을 클릭하고 **Advanced** 탭을 선택하여 선과 음영을 변경합니다. 각 그룹에서 선의 평균, 최소값 또는 최대값을 선택할 수 있습니다. 음영의 경우 음영을 끄고 최소값과 최대값, 표준 편차 및 표준 오차를 표시할 수 있습니다.

{{< img src="/images/track/demo_grouping_options_for_line_plots.gif" alt="" >}}
