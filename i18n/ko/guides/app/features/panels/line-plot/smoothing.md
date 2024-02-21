---
description: In line plots, use smoothing to see trends in noisy data.
displayed_sidebar: default
---

# 평활화

W&B 선 그래프에서는 세 가지 유형의 평활화를 지원합니다:

- [지수 이동 평균](smoothing.md#exponential-moving-average-default) (기본값)
- [가우시안 평활화](smoothing.md#gaussian-smoothing)
- [단순 이동 평균](smoothing.md#running-average)
- [지수 이동 평균 - Tensorboard](smoothing.md#exponential-moving-average-tensorboard) (사용 중단)

[대화형 W&B 리포트](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc)에서 실제로 확인하세요.

![](/images/app_ui/beamer_smoothing.gif)

## 지수 이동 평균 (기본값)

지수 평활화는 시간에 따른 데이터 평활화 기법으로, 이전 지점의 가중치를 지수적으로 감소시킵니다. 범위는 0에서 1입니다. 배경 지식은 [지수 평활화](https://www.wikiwand.com/en/Exponential_smoothing)를 참고하세요. 시계열의 초기 값이 0으로 편향되지 않도록 편향 제거 항이 추가됩니다.

EMA 알고리즘은 선상의 점 밀도(즉, x축 단위 범위당 `y` 값의 수)를 고려합니다. 이를 통해 동시에 다양한 특성을 가진 여러 라인을 표시할 때 일관된 평활화를 할 수 있습니다.

이것이 내부적으로 작동하는 방식의 샘플 코드입니다:

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

이것이 [앱에서](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc) 어떻게 보이는지 확인하세요:

![](/images/app_ui/weighted_exponential_moving_average.png)

## 가우시안 평활화

가우시안 평활화(또는 가우시안 커널 평활화)는 점들의 가중평균을 계산하는데, 가중치는 평활화 파라미터로 지정된 표준편차를 가진 가우시안 분포에 해당합니다. 모든 입력 x 값에 대해 평활화된 값이 계산됩니다.

TensorBoard의 동작과 일치시키는 것이 중요하지 않다면, 가우시안 평활화는 평활화를 위한 좋은 표준 선택입니다. 지수 이동 평균과 달리, 점은 값 이전과 이후에 발생하는 점들을 기반으로 평활화됩니다.

이것이 [앱에서](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#3.-gaussian-smoothing) 어떻게 보이는지 확인하세요:

![](/images/app_ui/gaussian_smoothing.png)

## 단순 이동 평균

단순 이동 평균은 주어진 x 값 이전과 이후의 창 내 점들의 평균으로 점을 대체하는 평활화 알고리즘입니다. 이동 평균에 대해 "박스카 필터"를 참고하세요 [https://en.wikipedia.org/wiki/Moving_average](https://en.wikipedia.org/wiki/Moving_average). 단순 이동 평균에 선택된 파라미터는 이동 평균을 고려할 점의 수를 Weights and Biases에 알려줍니다.

x축에서 점들이 균일하지 않게 배치되어 있다면 가우시안 평활화를 고려해 보세요.

다음 이미지는 [앱에서](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#4.-running-average) 단순 이동 평균이 어떻게 보이는지 보여줍니다:

![](/images/app_ui/running_average.png)

## 지수 이동 평균 (사용 중단)

> TensorBoard EMA 알고리즘은 x축 단위로 일관된 점 밀도(플롯된 점의 수)를 가지지 않는 동일한 차트의 여러 라인을 정확하게 평활화할 수 없기 때문에 사용이 중단되었습니다.

지수 이동 평균은 TensorBoard의 평활화 알고리즘과 일치하도록 구현되었습니다. 범위는 0에서 1입니다. 배경 지식은 [지수 평활화](https://www.wikiwand.com/en/Exponential_smoothing)를 참고하세요. 시계열의 초기 값이 0으로 편향되지 않도록 편향 제거 항이 추가되었습니다.

이것이 내부적으로 작동하는 방식의 샘플 코드입니다:

```javascript
  data.forEach(d => {
    const nextVal = d;
    last = last * smoothingWeight + (1 - smoothingWeight) * nextVal;
    numAccum++;
    debiasWeight = 1.0 - Math.pow(smoothingWeight, numAccum);
    smoothedData.push(last / debiasWeight);
```

이것이 [앱에서](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc) 어떻게 보이는지 확인하세요:

![](/images/app_ui/exponential_moving_average.png)

## 구현 세부 사항

모든 평활화 알고리즘은 샘플링된 데이터에서 실행됩니다. 즉, 1500개 이상의 점을 로그할 경우, 평활화 알고리즘은 서버에서 점들이 다운로드된 _이후에_ 실행됩니다. 평활화 알고리즘의 목적은 데이터에서 패턴을 빠르게 찾는 것을 돕는 것입니다. 많은 수의 로그된 점을 가진 메트릭에 대해 정확한 평활화된 값을 필요로 한다면 API를 통해 메트릭을 다운로드하고 자체 평활화 메서드를 실행하는 것이 좋습니다.

## 원본 데이터 숨기기

기본적으로 우리는 배경에서 원본, 평활화되지 않은 데이터를 희미한 선으로 보여줍니다. 이것을 끄려면 **Show Original** 토글을 클릭하세요.

![](/images/app_ui/demo_wandb_smoothing_turn_on_and_off_original_data.gif)