---
title: Smooth line plots
description: 노이즈가 많은 데이터에서 추세를 확인하기 위해 선형 플롯에서 스무딩을 사용하세요.
displayed_sidebar: default
---

W&B 라인 플롯에서는 세 가지 유형의 스무딩을 지원합니다:

- [지수 이동 평균](smoothing.md#exponential-moving-average-default) (기본값)
- [가우시안 스무딩](smoothing.md#gaussian-smoothing)
- [이동 평균](smoothing.md#running-average)
- [지수 이동 평균 - Tensorboard](smoothing.md#exponential-moving-average-deprecated) (폐기됨)

이 기능은 [인터랙티브 W&B 리포트](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc)에서 실시간으로 확인할 수 있습니다.

![](/images/app_ui/beamer_smoothing.gif)

## 지수 이동 평균 (기본값)

지수 스무딩은 이전 데이터 포인트의 가중치를 지수적으로 줄이며 시계열 데이터를 스무딩하는 기술입니다. 범위는 0에서 1입니다. 배경 지식을 위해 [Exponential Smoothing](https://www.wikiwand.com/en/Exponential_smoothing)를 참조하세요. 초기 시계열 값이 0으로 쏠리지 않도록 하는 디바이어스 항이 추가되었습니다.

EMA 알고리즘은 라인의 점 밀도 (즉, x축의 단위 범위당 `y` 값의 수)를 고려합니다. 이는 서로 다른 특성을 가진 여러 라인을 동시에 표시할 때 일관된 스무딩을 가능하게 합니다.

다음은 내부적으로 어떻게 작동하는지 보여주는 샘플 코드입니다:

```javascript
const smoothingWeight = Math.min(Math.sqrt(smoothingParam || 0), 0.999);
let lastY = yValues.length > 0 ? 0 : NaN;
let debiasWeight = 0;

return yValues.map((yPoint, index) => {
  const prevX = index > 0 ? index - 1 : 0;
  // VIEWPORT_SCALE은 결과를 차트의 x축 범위로 확장합니다
  const changeInX =
    ((xValues[index] - xValues[prevX]) / rangeOfX) * VIEWPORT_SCALE;
  const smoothingWeightAdj = Math.pow(smoothingWeight, changeInX);

  lastY = lastY * smoothingWeightAdj + yPoint;
  debiasWeight = debiasWeight * smoothingWeightAdj + 1;
  return lastY / debiasWeight;
});
```

다음은 [앱에서의 모습입니다](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

![](/images/app_ui/weighted_exponential_moving_average.png)

## 가우시안 스무딩

가우시안 스무딩(또는 가우시안 커널 스무딩)은 점들의 가중 평균을 계산하며, 가중치는 스무딩 파라미터로 지정된 표준 편차를 가진 가우시안 분포에 해당합니다. 스무딩된 값은 입력된 각 x 값에 대해 계산됩니다.

가우시안 스무딩은 TensorBoard의 행동과 일치시키는 것에 대해 걱정하지 않는다면 스무딩의 좋은 기본 선택입니다. 지수 이동 평균과 달리, 점은 값 앞뒤의 점을 기반으로 스무딩됩니다.

다음은 [앱에서의 모습입니다](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#3.-gaussian-smoothing):

![](/images/app_ui/gaussian_smoothing.png)

## 이동 평균

이동 평균은 특정 x 값 앞뒤의 포인트로 구성된 윈도우 내에서 포인트를 평균값으로 대체하는 스무딩 알고리즘입니다. [https://en.wikipedia.org/wiki/Moving_average](https://en.wikipedia.org/wiki/Moving_average)의 "박스카 필터"를 참조하세요. 이동 평균에서 선택된 파라미터는 Weights and Biases에 이동 평균에서 고려할 포인트의 수를 알려줍니다.

x축에 점이 불균등하게 간격을 두고 간격을 띄운 경우 가우시안 스무딩을 사용하는 것을 고려하세요.

다음 이미지는 [앱에서의 모습입니다](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#4.-running-average):

![](/images/app_ui/running_average.png)

## 지수 이동 평균 (폐기됨)

> TensorBoard EMA 알고리즘은 동일한 차트에서 일관된 점 밀도를 가지지 않는 여러 라인을 정확하게 스무딩할 수 없으므로 폐기되었습니다.

지수 이동 평균은 TensorBoard의 스무딩 알고리즘과 일치하도록 구현되었습니다. 범위는 0에서 1입니다. 배경 지식을 위해 [Exponential Smoothing](https://www.wikiwand.com/en/Exponential_smoothing)를 참조하세요. 초기 시계열 값이 0으로 바이어스되지 않도록 하는 디바이어스 항이 추가되었습니다.

다음은 내부적으로 어떻게 작동하는지 보여주는 샘플 코드입니다:

```javascript
  data.forEach(d => {
    const nextVal = d;
    last = last * smoothingWeight + (1 - smoothingWeight) * nextVal;
    numAccum++;
    debiasWeight = 1.0 - Math.pow(smoothingWeight, numAccum);
    smoothedData.push(last / debiasWeight);
```

다음은 [앱에서의 모습입니다](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

![](/images/app_ui/exponential_moving_average.png)

## 구현 세부사항

모든 스무딩 알고리즘은 샘플링된 데이터에서 실행됩니다. 즉, 1500개 이상의 포인트가 로그되면 스무딩 알고리즘은 서버에서 포인트가 다운로드된 후 실행됩니다. 스무딩 알고리즘의 목적은 데이터를 빠르게 확인하여 패턴을 찾는 것입니다. 수많은 로그된 포인트를 가진 메트릭에 대한 정확한 스무딩 값을 필요로 한다면, API를 통해 메트릭을 다운로드하고 사용자만의 스무딩 메소드를 실행하는 것이 더 나을 수 있습니다.

## 원본 데이터 숨기기

기본적으로 우리는 원본, 스무딩되지 않은 데이터를 백그라운드에 희미한 선으로 표시합니다. **Show Original** 토글을 클릭하여 이를 끌 수 있습니다.

![](/images/app_ui/demo_wandb_smoothing_turn_on_and_off_original_data.gif)