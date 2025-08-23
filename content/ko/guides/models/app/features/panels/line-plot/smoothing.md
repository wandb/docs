---
title: 부드러운 선 그래프
description: 라인 플롯에서는 스무딩을 사용하여 노이즈가 많은 데이터에서 트렌드를 확인할 수 있습니다.
menu:
  default:
    identifier: ko-guides-models-app-features-panels-line-plot-smoothing
    parent: line-plot
weight: 30
---

W&B는 여러 가지 smoothing 방법을 지원합니다:

- [시간 가중 지수 이동 평균(TWEMA) 스무딩]({{< relref path="#time-weighted-exponential-moving-average-twema-smoothing-default" lang="ko" >}})
- [가우시안 스무딩]({{< relref path="#gaussian-smoothing" lang="ko" >}})
- [러닝 에버리지 스무딩]({{< relref path="#running-average-smoothing" lang="ko" >}})
- [지수 이동 평균(EMA) 스무딩]({{< relref path="#exponential-moving-average-ema-smoothing" lang="ko" >}})

[인터랙티브 W&B Report](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc)에서 실제 작동하는 모습을 확인할 수 있습니다.

{{< img src="/images/app_ui/beamer_smoothing.gif" alt="다양한 스무딩 알고리즘 데모" >}}

## 시간 가중 지수 이동 평균(TWEMA) 스무딩 (기본값)

시간 가중 지수 이동 평균(TWEMA) 스무딩 알고리즘은 이전 데이터의 가중치를 지수적으로 감소시키며 시계열 데이터를 부드럽게 만드는 방식입니다. 자세한 기법 설명은 [Exponential Smoothing](https://www.wikiwand.com/en/Exponential_smoothing)을 참고하세요. 범위는 0에서 1 사이이며, 시계열의 초반 값들이 0으로 치우치지 않도록 디바이어스(de-bias) 항이 추가되어 있습니다.

TWEMA 알고리즘은 선 위의 포인트 밀도(`y` 값이 x축의 범위 단위당 몇 개 있는지)를 고려합니다. 이를 통해 특성이 다른 여러 선을 동시에 표시할 때 일관된 smoothing이 가능합니다.

아래는 TWEMA가 내부적으로 동작하는 방법의 예시 코드입니다:

```javascript
const smoothingWeight = Math.min(Math.sqrt(smoothingParam || 0), 0.999);
let lastY = yValues.length > 0 ? 0 : NaN;
let debiasWeight = 0;

return yValues.map((yPoint, index) => {
  const prevX = index > 0 ? index - 1 : 0;
  // VIEWPORT_SCALE: 결과를 차트의 x축 범위에 맞게 스케일링합니다.
  const changeInX =
    ((xValues[index] - xValues[prevX]) / rangeOfX) * VIEWPORT_SCALE;
  const smoothingWeightAdj = Math.pow(smoothingWeight, changeInX);

  lastY = lastY * smoothingWeightAdj + yPoint;
  debiasWeight = debiasWeight * smoothingWeightAdj + 1;
  return lastY / debiasWeight;
});
```

[앱에서 실제 예시를 확인하세요.](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

{{< img src="/images/app_ui/weighted_exponential_moving_average.png" alt="TWEMA 스무딩 데모" >}}

## 가우시안 스무딩

가우시안 스무딩(또는 가우시안 커널 스무딩)은 각 포인트의 가중 평균을 계산하는데, 여기서 가중치는 지정한 표준편차를 가진 가우시안 분포를 따릅니다. 각 입력 x 값마다 스무딩된 값을 계산하며, 해당 x의 이전과 이후에 위치한 포인트 모두를 반영합니다.

[앱에서 실제 예시를 확인하세요.](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#3.-gaussian-smoothing):

{{< img src="/images/app_ui/gaussian_smoothing.png" alt="가우시안 스무딩 데모" >}}

## 러닝 에버리지 스무딩

러닝 에버리지는 주어진 x 값 이전과 이후 윈도우 안의 포인트 평균으로 한 포인트를 대체하는 smoothing 알고리즘입니다. 자세한 내용은 ["Boxcar Filter" (Wikipedia)](https://en.wikipedia.org/wiki/Moving_average)를 참고하세요. 러닝 에버리지 파라미터는 이동 평균을 계산할 때 고려할 포인트 수를 지정합니다.

x축상 포인트 간격이 일정하지 않다면, 가우시안 스무딩을 사용하는 것이 더 적합할 수 있습니다.

[앱에서 실제 예시를 확인하세요.](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc#4.-running-average):

{{< img src="/images/app_ui/running_average.png" alt="러닝 에버리지 스무딩 데모" >}}

## 지수 이동 평균(EMA) 스무딩

지수 이동 평균(EMA) 스무딩 알고리즘은 지수 윈도우 함수를 사용하여 시계열 데이터를 부드럽게 만드는, 널리 쓰이는 방법입니다. 자세한 설명은 [Exponential Smoothing](https://www.wikiwand.com/en/Exponential_smoothing)을 참고하세요. 범위는 0~1이며, 시계열 초반 값들이 0 쪽으로 치우치지 않게 디바이어스 항이 추가됩니다.

많은 경우 EMA 스무딩은 데이터를 먼저 버킷팅하지 않고 전체 기록에 대해 직접 적용됩니다. 이럴 때 smoothing의 정확도가 더 높게 나옵니다.

반대로, 아래와 같은 경우에는 버킷팅을 거친 후 EMA smoothing이 적용됩니다:
- 샘플링
- 그룹핑
- 수식 계산
- 비단조적(non-monotonic) x축
- 시간 기반 x축

아래는 EMA가 내부적으로 동작하는 방법의 예시 코드입니다:

```javascript
  data.forEach(d => {
    const nextVal = d;
    last = last * smoothingWeight + (1 - smoothingWeight) * nextVal;
    numAccum++;
    debiasWeight = 1.0 - Math.pow(smoothingWeight, numAccum);
    smoothedData.push(last / debiasWeight);
```

[앱에서 실제 예시를 확인하세요.](https://wandb.ai/carey/smoothing-example/reports/W-B-Smoothing-Features--Vmlldzo1MzY3OTc):

{{< img src="/images/app_ui/exponential_moving_average.png" alt="EMA 스무딩 데모" >}}

## 원본 데이터 숨기기

기본적으로, 스무딩 전 원본 데이터는 플롯의 뒷배경에 흐리게 표시됩니다. **Show Original** 버튼을 클릭하면 이 표시를 끌 수 있습니다.

{{< img src="/images/app_ui/demo_wandb_smoothing_turn_on_and_off_original_data.gif" alt="원본 데이터 표시/숨김 전환" >}}