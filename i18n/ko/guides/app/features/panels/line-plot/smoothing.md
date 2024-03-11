---
description: In line plots, use smoothing to see trends in noisy data.
displayed_sidebar: default
---

# 스무딩

W&B 라인 플롯에서는 다음 세 가지 유형의 스무딩을 지원합니다:

- [지수 이동 평균](smoothing.md#exponential-moving-average-default) (기본값)
- [가우시안 스무딩](smoothing.md#gaussian-smoothing)
- [단순 이동 평균](smoothing.md#running-average)
- [지수 이동 평균 - Tensorboard](smoothing.md#exponential-moving-average-tensorboard) (사용 중단됨)

[대화형 W&B 리포트](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc)에서 실시간으로 확인하세요.

![](/images/app_ui/beamer_smoothing.gif)

## 지수 이동 평균 (기본값)

지수 스무딩은 시간에 따른 데이터를 이전 데이터 포인트의 가중치를 지수적으로 감소시키면서 스무딩하는 기법입니다. 범위는 0에서 1입니다. 배경 지식은 [지수 스무딩](https://www.wikiwand.com/en/Exponential_smoothing)에서 확인하세요. 시계열의 초기 값이 0을 향해 편향되지 않도록 비편향 항이 추가됩니다.

EMA 알고리즘은 선 위의 점의 밀도(즉, x축 단위 범위당 `y` 값의 개수)를 고려합니다. 이를 통해 동시에 서로 다른 특성을 가진 여러 라인을 일관되게 스무딩할 수 있습니다.

여기 내부에서 이것이 어떻게 작동하는지에 대한 샘플 코드가 있습니다:

```javascript
const smoothingWeight = Math.min(Math.sqrt(smoothingParam || 0), 0.999);
let lastY = yValues.length > 0 ? 0 : NaN;
let debiasWeight = 0;

return yValues.map((yPoint, index) => {
  const prevX = index > 0 ? index - 1 : 0;
  // VIEWPORT_SCALE은 결과를 차트의 x축 범위에 맞게 조정합니다
  const changeInX =
    ((xValues[index] - xValues[prevX]) / rangeOfX) * VIEWPORT_SCALE;
  const smoothingWeightAdj = Math.pow(smoothingWeight, changeInX);

  lastY = lastY * smoothingWeightAdj + yPoint;
  debiasWeight = debiasWeight * smoothingWeightAdj + 1;
  return lastY / debiasWeight;
});
```

[앱에서](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc) 이것이 어떻게 보이는지 여기 있습니다:

![](/images/app_ui/weighted_exponential_moving_average.png)

## 가우시안 스무딩

가우시안 스무딩(또는 가우시안 커널 스무딩)은 점들의 가중 평균을 계산하는데, 가중치는 스무딩 파라미터로 지정된 표준 편차를 가진 가우시안 분포에 해당합니다. 스무딩된 값은 모든 입력 x 값에 대해 계산됩니다.

가우시안 스무딩은 TensorBoard의 행동을 일치시키는 것에 대해 걱정하지 않는다면 스무딩을 위한 좋은 표준 선택입니다. 지수 이동 평균과 달리 점은 값 이전 및 이후에 발생하는 점에 기반하여 스무딩됩니다.

[앱에서](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#3.-gaussian-smoothing) 이것이 어떻게 보이는지 여기 있습니다:

![](/images/app_ui/gaussian_smoothing.png)

## 단순 이동 평균

단순 이동 평균은 주어진 x 값 이전 및 이후의 창 내 점들의 평균으로 점을 대체하는 스무딩 알고리즘입니다. "Boxcar Filter"를 참조하세요 [https://en.wikipedia.org/wiki/Moving_average](https://en.wikipedia.org/wiki/Moving_average). 단순 이동 평균에 선택된 파라미터는 이동 평균에서 고려하는 점의 수를 Weights and Biases에 알려줍니다.

x축에 점들이 고르지 않게 배치된 경우 가우시안 스무딩을 사용하는 것이 좋습니다.

다음 이미지는 [앱에서](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#4.-running-average) 달리기 앱이 어떻게 보이는지 보여줍니다:

![](/images/app_ui/running_average.png)

## 지수 이동 평균 (사용 중단됨)

> TensorBoard EMA 알고리즘은 x축 단위로 일관된 점 밀도(플롯된 점의 수)를 가지지 않는 동일 차트 상의 여러 라인을 정확하게 스무딩할 수 없기 때문에 사용 중단되었습니다.

지수 이동 평균은 TensorBoard의 스무딩 알고리즘과 일치하게 구현됩니다. 범위는 0에서 1입니다. 배경 지식은 [지수 스무딩](https://www.wikiwand.com/en/Exponential_smoothing)에서 확인하세요. 시계열의 초기 값이 0을 향해 편향되지 않도록 비편향 항이 추가되었습니다.

여기 내부에서 이것이 어떻게 작동하는지에 대한 샘플 코드가 있습니다:

```javascript
  data.forEach(d => {
    const nextVal = d;
    last = last * smoothingWeight + (1 - smoothingWeight) * nextVal;
    numAccum++;
    debiasWeight = 1.0 - Math.pow(smoothingWeight, numAccum);
    smoothedData.push(last / debiasWeight);
```

[앱에서](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc) 이것이 어떻게 보이는지 여기 있습니다:

![](/images/app_ui/exponential_moving_average.png)

## 구현 세부 사항

모든 스무딩 알고리즘은 샘플링된 데이터에서 실행되며, 즉 1500개 이상의 점을 로그하면 스무딩 알고리즘은 서버에서 점들이 다운로드된 _후에_ 실행됩니다. 스무딩 알고리즘의 의도는 데이터에서 빠르게 패턴을 찾는 데 도움을 주는 것입니다. 많은 수의 로그된 점을 가진 메트릭에서 정확한 스무딩된 값을 필요로 한다면, API를 통해 메트릭을 다운로드하고 자체 스무딩 메소드를 실행하는 것이 더 좋을 수 있습니다.

## 원본 데이터 숨기기

기본적으로 우리는 배경에서 희미한 선으로 원본, 스무딩되지 않은 데이터를 보여줍니다. 이를 끄려면 **Show Original** 토글을 클릭하세요.

![](/images/app_ui/demo_wandb_smoothing_turn_on_and_off_original_data.gif)