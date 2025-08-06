---
title: 라인 플롯 참고 문서
menu:
  default:
    identifier: ko-guides-models-app-features-panels-line-plot-reference
    parent: line-plot
weight: 10
---

## X-Axis

{{< img src="/images/app_ui/reference_x_axis.png" alt="Selecting X-Axis" >}}

선형 플롯의 x축은 W&B.log로 숫자 값으로 기록된 모든 값으로 설정할 수 있습니다.

## Y-Axis 변수

y축 변수는 wandb.log로 숫자, 숫자 배열 또는 숫자 히스토그램을 기록했다면 어떤 값이든 설정할 수 있습니다. 만약 하나의 변수에 대해 1500포인트 이상을 기록했다면, W&B가 1500포인트로 샘플링해서 보여줍니다.

{{% alert %}}
y축 선의 색상은 runs 테이블에서 해당 run의 색상을 변경함으로써 바꿀 수 있습니다.
{{% /alert %}}

## X 범위 및 Y 범위

플롯의 X와 Y의 최대/최소 값을 조정할 수 있습니다.

X 범위의 기본값은 x축에 기록된 값 중 가장 작은 값부터 가장 큰 값까지입니다.

Y 범위의 기본값은 메트릭의 최솟값과 0, 그리고 메트릭의 최댓값까지입니다.

## 최대 runs/그룹

기본적으로 한 번에 10개의 run 또는 run 그룹만 플로팅됩니다. 표시되는 run은 runs 테이블 또는 run 세트 상단에서 가져오므로, runs 테이블이나 run 세트를 정렬하면 표시되는 run을 변경할 수 있습니다.

{{% alert %}}
워크스페이스에서는 설정과 상관없이 최대 1000개의 run만 표시할 수 있습니다.
{{% /alert %}}

## 범례(Legend)

차트의 범례에서 각 run별로 기록된 config 값이나 run의 메타 데이터(예: 생성 시각, run 생성자)를 표시할 수 있습니다.

예시:

`${run:displayName} - ${config:dropout}`로 설정하면 각 run의 범례명이 `royal-sweep - 0.5`처럼 표시됩니다. 여기서 `royal-sweep`은 run 이름이고 `0.5`는 `dropout`이라는 설정 파라미터 입니다.

차트 위에 마우스를 올리면 크로스헤어에 포인트별 값을 보여주기 위해 `[[ ]]` 내부에 값을 설정할 수 있습니다. 예를 들어 `\[\[ $x: $y ($original) ]]`로 입력하면 "2: 3 (2.9)"와 같이 표시됩니다.

`[[ ]]` 내부에서 사용할 수 있는 값들은 다음과 같습니다:

| 값           | 의미                                         |
| ------------ | ------------------------------------------- |
| `${x}`        | X 값                                      |
| `${y}`        | Y 값 (스무딩 적용 포함)                    |
| `${original}` | 스무딩이 적용되지 않은 Y 값                |
| `${mean}`     | 그룹화된 run의 평균                        |
| `${stddev}`   | 그룹화된 run의 표준편차                    |
| `${min}`      | 그룹화된 run 중 최소값                     |
| `${max}`      | 그룹화된 run 중 최대값                     |
| `${percent}`  | 전체에서의 비율(누적 면적 그래프용)        |

## 그룹화(Grouping)

그룹화를 켜면 전체 run을 집계하거나, 선택한 개별 변수로 그룹화할 수 있습니다. 테이블 내에서 그룹화를 설정하면, 해당 그룹 정보가 자동으로 그래프에도 반영됩니다.

## 스무딩(Smoothing)

[스무딩 계수]({{< relref path="/support/kb-articles/formula_smoothing_algorithm.md" lang="ko" >}})는 0과 1 사이로 설정할 수 있고, 0은 스무딩 없음, 1은 최대 스무딩입니다.

## 이상치 무시

이상치를 기본 플롯의 최소/최대 범위에서 제외하여 차트를 재스케일링할 수 있습니다. 이 설정의 효과는 플롯의 샘플링 모드에 따라 달라집니다.

- [무작위 샘플링 모드]({{< relref path="sampling.md#random-sampling" lang="ko" >}})에서는 **이상치 무시**를 켜면 5%에서 95% 구간의 포인트만 표시됩니다. 이상치가 표시될 때 다른 포인트와 구별되지는 않습니다.
- [전체 정밀도 모드]({{< relref path="sampling.md#full-fidelity" lang="ko" >}})에서는 모든 포인트가 항상 표시되며, 각 버킷의 마지막 값으로 압축됩니다. **이상치 무시**를 켜면 각 버킷의 최소/최대 경계가 음영처리 됩니다. 비활성화 시에는 음영처리 없이 표시됩니다.

## 수식(Expression)

수식(Expression) 기능을 이용해 1-accuracy 처럼 메트릭 값을 조합한 결과를 플롯할 수 있습니다. 현재는 하나의 메트릭만 플롯할 때 동작합니다. 기본적인 산술 연산자 (+, -, \*, /, %, 그리고 \*\* 거듭제곱)도 사용할 수 있습니다.

## 플롯 스타일

선형 플롯의 스타일을 선택할 수 있습니다.

**라인 플롯(Line plot):**

{{< img src="/images/app_ui/plot_style_line_plot.png" alt="Line plot style" >}}

**면적 플롯(Area plot):**

{{< img src="/images/app_ui/plot_style_area_plot.png" alt="Area plot style" >}}

**백분율 면적 플롯(Percentage area plot):**

{{< img src="/images/app_ui/plot_style_percentage_plot.png" alt="Percentage plot style" >}}