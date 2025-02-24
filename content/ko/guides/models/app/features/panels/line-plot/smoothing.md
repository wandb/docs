---
title: Smooth line plots
description: 꺾은선 그래프에서 스무딩을 사용하면 노이즈가 많은 데이터에서 추세를 확인할 수 있습니다.
menu:
  default:
    identifier: ko-guides-models-app-features-panels-line-plot-smoothing
    parent: line-plot
weight: 30
---

W&B는 세 가지 유형의 스무딩을 지원합니다:

- [지수 이동 평균]({{< relref path="smoothing.md#exponential-moving-average-default" lang="ko" >}}) (기본값)
- [가우시안 스무딩]({{< relref path="smoothing.md#gaussian-smoothing" lang="ko" >}})
- [이동 평균]({{< relref path="smoothing.md#running-average" lang="ko" >}})
- [지수 이동 평균 - Tensorboard]({{< relref path="smoothing.md#exponential-moving-average-deprecated" lang="ko" >}}) (더 이상 사용되지 않음)

[대화형 W&B 리포트](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc)에서 이를 라이브로 확인하세요.

{{< img src="/images/app_ui/beamer_smoothing.gif" alt="" >}}

## 지수 이동 평균 (기본값)

지수 스무딩은 이전 포인트의 가중치를 지수적으로 감쇠시켜 시계열 데이터를 스무딩하는 기술입니다. 범위는 0에서 1 사이입니다. 배경 정보는 [지수 스무딩](https://www.wikiwand.com/en/Exponential_smoothing)을 참조하세요. 시계열의 초기 값이 0으로 치우치지 않도록 디바이어스 항이 추가되었습니다.

EMA 알고리즘은 선의 포인트 밀도 (x축 범위 단위당 `y` 값의 수)를 고려합니다. 이를 통해 특성이 다른 여러 라인을 동시에 표시할 때 일관된 스무딩이 가능합니다.

다음은 내부 작동 방식에 대한 샘플 코드입니다:

```javascript
const smoothingWeight = Math.min(Math.sqrt(smoothingParam || 0), 0.999);
let lastY = yValues.length > 0 ? 0 : NaN;
let debiasWeight = 0;

return yValues.map((yPoint, index) => {
  const prevX = index > 0 ? index - 1 : 0;
  // VIEWPORT_SCALE scales the result to the chart's x-axis range
  const changeInX =
    ((xValues[index] - xValues[prevX]) / rangeOfX) * VIEWPORT_SCALE;
  const smoothingWeightAdj = Math.pow(smoothingWeight, changeInX);

  lastY = lastY * smoothingWeightAdj + yPoint;
  debiasWeight = debiasWeight * smoothingWeightAdj + 1;
  return lastY / debiasWeight;
});
```

[앱에서](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc) 다음과 같이 표시됩니다:

{{< img src="/images/app_ui/weighted_exponential_moving_average.png" alt="" >}}

## 가우시안 스무딩

가우시안 스무딩 (또는 가우시안 커널 스무딩)은 가중 평균을 계산합니다. 여기서 가중치는 스무딩 파라미터로 지정된 표준 편차를 갖는 가우시안 분포에 해당합니다. 스무딩된 값은 모든 입력 x 값에 대해 계산됩니다.

TensorBoard의 동작과 일치시키는 데 관심이 없다면 가우시안 스무딩은 스무딩을 위한 좋은 표준 선택입니다. 지수 이동 평균과 달리 포인트는 값 이전과 이후에 발생하는 포인트를 기반으로 스무딩됩니다.

[앱에서](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#3.-gaussian-smoothing) 다음과 같이 표시됩니다:

{{< img src="/images/app_ui/gaussian_smoothing.png" alt="" >}}

## 이동 평균

이동 평균은 주어진 x 값 전후의 창에서 포인트를 평균으로 대체하는 스무딩 알고리즘입니다. [https://en.wikipedia.org/wiki/Moving_average](https://en.wikipedia.org/wiki/Moving_average)의 "Boxcar Filter"를 참조하세요. 이동 평균에 대해 선택된 파라미터는 Weights and Biases에 이동 평균에서 고려할 포인트 수를 알려줍니다.

포인트가 x축에서 고르지 않게 배치된 경우 가우시안 스무딩을 사용하는 것이 좋습니다.

다음 이미지는 이동 앱이 [앱에서](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#4.-running-average) 어떻게 보이는지 보여줍니다:

{{< img src="/images/app_ui/running_average.png" alt="" >}}

## 지수 이동 평균 (더 이상 사용되지 않음)

> TensorBoard EMA 알고리즘은 포인트 밀도가 일관되지 않은 (x축 단위당 플롯된 포인트 수) 동일한 차트에서 여러 라인을 정확하게 스무딩할 수 없으므로 더 이상 사용되지 않습니다.

지수 이동 평균은 TensorBoard의 스무딩 알고리즘과 일치하도록 구현됩니다. 범위는 0에서 1 사이입니다. 배경 정보는 [지수 스무딩](https://www.wikiwand.com/en/Exponential_smoothing)을 참조하세요. 시계열의 초기 값이 0으로 치우치지 않도록 디바이어스 항이 추가되었습니다.

다음은 내부 작동 방식에 대한 샘플 코드입니다:

```javascript
  data.forEach(d => {
    const nextVal = d;
    last = last * smoothingWeight + (1 - smoothingWeight) * nextVal;
    numAccum++;
    debiasWeight = 1.0 - Math.pow(smoothingWeight, numAccum);
    smoothedData.push(last / debiasWeight);
```

[앱에서](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc) 다음과 같이 표시됩니다:

{{< img src="/images/app_ui/exponential_moving_average.png" alt="" >}}

## 구현 세부 정보

모든 스무딩 알고리즘은 샘플링된 데이터에서 실행됩니다. 즉, 1500개 이상의 포인트를 기록하면 스무딩 알고리즘은 포인트가 서버에서 다운로드된 _후에_ 실행됩니다. 스무딩 알고리즘의 목적은 데이터에서 패턴을 빠르게 찾는 데 도움을 주는 것입니다. 많은 수의 기록된 포인트가 있는 메트릭에서 정확한 스무딩된 값이 필요한 경우 API를 통해 메트릭을 다운로드하고 자체 스무딩 methods를 실행하는 것이 좋습니다.

## 원본 데이터 숨기기

기본적으로 배경에 흐릿한 선으로 스무딩되지 않은 원본 데이터가 표시됩니다. 이를 끄려면 **원본 보기** 토글을 클릭하세요.

{{< img src="/images/app_ui/demo_wandb_smoothing_turn_on_and_off_original_data.gif" alt="" >}}
