---
title: Line plot reference
menu:
  default:
    identifier: ko-guides-models-app-features-panels-line-plot-reference
    parent: line-plot
weight: 10
---

## X축

{{< img src="/images/app_ui/reference_x_axis.png" alt="Selecting X-Axis" >}}

W&B.log 로 로그한 값이라면, 항상 숫자로 기록되는 한, 라인 플롯의 x축을 어떤 값으로든 설정할 수 있습니다.

## Y축 변수

wandb.log 로 로그한 값이라면, 숫자, 숫자 배열 또는 숫자 히스토그램을 로깅하는 한, y축 변수를 어떤 값으로든 설정할 수 있습니다. 변수에 대해 1500개 이상의 포인트를 로깅한 경우, W&B 는 1500개 포인트로 샘플링합니다.

{{% alert %}}
runs 테이블에서 run의 색상을 변경하여 y축 선의 색상을 변경할 수 있습니다.
{{% /alert %}}

## X 범위 및 Y 범위

플롯의 X 및 Y의 최대값과 최소값을 변경할 수 있습니다.

X 범위의 기본값은 x축의 최소값에서 최대값까지입니다.

Y 범위의 기본값은 메트릭의 최소값 및 0에서 메트릭의 최대값까지입니다.

## 최대 Runs/Groups

기본적으로 10개의 run 또는 run 그룹만 플롯됩니다. run은 runs 테이블 또는 run 세트의 맨 위에서 가져오므로 runs 테이블 또는 run 세트를 정렬하면 표시되는 run을 변경할 수 있습니다.

## 범례

차트의 범례를 제어하여 생성 시간 또는 run을 생성한 user 와 같은 run의 메타 데이터 및 기록한 모든 구성 값을 모든 run에 대해 표시할 수 있습니다.

예:

`${run:displayName} - ${config:dropout}` 은 각 run의 범례 이름을 `royal-sweep - 0.5` 와 같이 만듭니다. 여기서 `royal-sweep` 은 run 이름이고 `0.5` 는 `dropout` 이라는 구성 파라미터입니다.

차트 위로 마우스를 가져갈 때 십자선에 포인트 특정 값을 표시하려면 `[[ ]]` 안에 값을 설정할 수 있습니다. 예를 들어 `\[\[ $x: $y ($original) ]]` 은 "2: 3 (2.9)" 와 같이 표시합니다.

`[[ ]]` 내에서 지원되는 값은 다음과 같습니다.

| 값             | 의미                                           |
| --------------- | --------------------------------------------- |
| `${x}`          | X 값                                          |
| `${y}`          | Y 값(평활 조정 포함)                            |
| `${original}`   | Y 값(평활 조정 미포함)                         |
| `${mean}`       | 그룹화된 run의 평균                             |
| `${stddev}`     | 그룹화된 run의 표준 편차                          |
| `${min}`        | 그룹화된 run의 최소값                            |
| `${max}`        | 그룹화된 run의 최대값                            |
| `${percent}`    | 총계의 백분율(스택 영역 차트의 경우)              |

## 그룹화

그룹화를 켜서 모든 run을 집계하거나 개별 변수별로 그룹화할 수 있습니다. 테이블 내에서 그룹화하여 그룹화를 켤 수도 있으며 그룹이 그래프에 자동으로 채워집니다.

## 평활

[평활 계수]({{< relref path="/support/formula_smoothing_algorithm.md" lang="ko" >}}) 를 0과 1 사이로 설정할 수 있습니다. 여기서 0은 평활이 없고 1은 최대 평활입니다.

## 이상값 무시

기본 플롯 최소 및 최대 스케일에서 이상값을 제외하도록 플롯의 스케일을 조정합니다. 플롯에 대한 설정의 영향은 플롯의 샘플링 모드에 따라 달라집니다.

- [임의 샘플링 모드]({{< relref path="sampling.md#random-sampling" lang="ko" >}}) 를 사용하는 플롯의 경우 **이상값 무시** 를 활성화하면 5%에서 95%의 포인트만 표시됩니다. 이상값이 표시되면 다른 포인트와 다른 방식으로 포맷되지 않습니다.
- [전체 충실도 모드]({{< relref path="sampling.md#full-fidelity" lang="ko" >}}) 를 사용하는 플롯의 경우 모든 포인트가 항상 표시되며 각 버킷의 마지막 값으로 축소됩니다. **이상값 무시** 를 활성화하면 각 버킷의 최소 및 최대 경계가 음영 처리됩니다. 그렇지 않으면 영역이 음영 처리되지 않습니다.

## 표현식

표현식을 사용하면 1-정확도와 같은 메트릭에서 파생된 값을 플롯할 수 있습니다. 현재 단일 메트릭을 플롯하는 경우에만 작동합니다. 간단한 산술 표현식, +, -, \*, / 및 % 와 ** (거듭제곱) 을 사용할 수 있습니다.

## 플롯 스타일

라인 플롯의 스타일을 선택합니다.

**라인 플롯:**

{{< img src="/images/app_ui/plot_style_line_plot.png" alt="" >}}

**영역 플롯:**

{{< img src="/images/app_ui/plot_style_area_plot.png" alt="" >}}

**백분율 영역 플롯:**

{{< img src="/images/app_ui/plot_style_percentage_plot.png" alt="" >}}
