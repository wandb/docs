---
title: Line plot reference
displayed_sidebar: default
---

## X-Axis

![Selecting X-Axis](/images/app_ui/reference_x_axis.png)

선 그래프의 X-Axis를 wandb.log로 기록한 숫자 값으로 설정할 수 있습니다.

## Y-Axis Variables

wandb.log로 기록한 숫자, 숫자 배열 또는 숫자의 히스토그램 값을 y-axis 변수로 설정할 수 있습니다. 만약 하나의 변수에 대해 1500점 이상을 기록했다면, wandb는 이를 1500점으로 샘플링합니다.

:::info
runs 테이블에서 run의 색상을 변경하여 y-axis 선의 색상을 바꿀 수 있습니다.
:::

## X Range and Y Range

그래프의 X와 Y의 최대값과 최소값을 변경할 수 있습니다.

X 범위의 기본값은 x-axis 값 중 가장 작은 값에서 가장 큰 값까지입니다.

Y 범위의 기본값은 메트릭의 가장 작은 값과 0에서 메트릭의 가장 큰 값까지입니다.

## Max Runs/Groups

기본적으로 10개의 runs 또는 그룹만을 플롯합니다. runs는 runs 테이블이나 run 세트의 상단에서 가져오며, runs 테이블이나 run 세트를 정렬하여 표시되는 runs를 변경할 수 있습니다.

## Legend

차트의 범례를 제어하여 기록한 모든 config 값과 run의 메타 데이터(생성 시간, run을 생성한 사용자 등)를 표시할 수 있습니다.

예시:

`${run:displayName} - ${config:dropout}`는 각 run에 대한 범례 이름을 `royal-sweep - 0.5`처럼 설정합니다. 여기서 `royal-sweep`은 run 이름이고 `0.5`는 `dropout`이라는 config 파라미터입니다.

`[[ ]]` 안에 값을 설정하여 차트를 가리킬 때마다 크로스헤어에서 특정 포인트 값을 표시할 수 있습니다. 예를 들어 `\[\[ $x: $y ($original) ]]`는 "2: 3 (2.9)"와 같은 값을 표시합니다.

`[[ ]]` 안에서 지원되는 값은 다음과 같습니다:

| 값           | 의미                                         |
| ------------  | ------------------------------------------ |
| `${x}`        | X 값                                       |
| `${y}`        | Y 값 (스무딩 조정 포함)                    |
| `${original}` | 스무딩 조정을 포함하지 않은 Y 값            |
| `${mean}`     | 그룹된 runs 의 평균                         |
| `${stddev}`   | 그룹된 runs 의 표준 편차                    |
| `${min}`      | 그룹된 runs 의 최소 값                       |
| `${max}`      | 그룹된 runs 의 최대 값                       |
| `${percent}`  | 총 비율 (누적 영역 차트의 경우)             |

## Grouping

그룹핑을 활성화하여 모든 runs를 집계하거나 개별 변수로 그룹화할 수 있습니다. 테이블에서 그룹핑하여 그래프로 자동 반영되도록 할 수도 있습니다.

## Smoothing

[smoothing coefficient](../../../../technical-faq/general.md#what-formula-do-you-use-for-your-smoothing-algorithm)을 0과 1 사이의 값으로 설정할 수 있으며, 0은 스무딩이 없고 1은 최대 스무딩입니다.

## Ignore Outliers

이상치를 무시하면 그래프에서 y-axis의 최소값과 최대값을 데이터의 5번째와 95번째 백분위로 설정하여 모든 데이터를 보이도록 설정하지 않습니다.

## Expression

Expression을 사용하여 1-accuracy와 같은 메트릭에서 파생된 값을 플롯할 수 있습니다. 현재는 단일 메트릭만 플롯할 때 작동합니다. 간단한 산술 표현식, +, -, \*, / 와 % 뿐만 아니라 거듭제곱을 위한 \*\* 도 사용할 수 있습니다.

## Plot style

선 그래프의 스타일을 선택하세요.

**Line plot:**

![](/images/app_ui/plot_style_line_plot.png)

**Area plot:**

![](/images/app_ui/plot_style_area_plot.png)

**Percentage area plot:**

![](/images/app_ui/plot_style_percentage_plot.png)