---
title: 산점도
menu:
  default:
    identifier: ko-guides-models-app-features-panels-scatter-plot
    parent: panels
weight: 40
---

이 페이지에서는 W&B에서 산점도를 사용하는 방법을 안내합니다.

## 유스 케이스

산점도를 사용하여 여러 run 을 비교하고 실험의 성능을 시각화할 수 있습니다.

- 최소, 최대, 평균 값을 선으로 표시합니다.
- 메타데이터 툴팁을 맞춤 설정합니다.
- 포인트 색상을 조절할 수 있습니다.
- 축의 범위를 조정할 수 있습니다.
- 축에 로그 스케일을 적용할 수 있습니다.

## 예시

아래 예시는 여러 주에 걸친 실험에서 다양한 모델의 검증 정확도를 보여주는 산점도입니다. 툴팁에는 배치 크기, 드롭아웃, 축 값이 포함되어 있습니다. 실행 평균 검증 정확도를 나타내는 선도 표시됩니다.

[라이브 예시 보기 →](https://app.wandb.ai/l2k2/l2k/reports?view=carey%2FScatter%20Plot)

{{< img src="/images/general/scatter-plots-1.png" alt="Validation accuracy scatter plot" >}}

## 산점도 만들기

W&B UI에서 산점도를 만들려면 다음 단계를 따르세요.

1. **Workspaces** 탭으로 이동합니다.
2. **Charts** 패널에서 액션 메뉴 `...` 를 클릭합니다.
3. 팝업 메뉴에서 **Add panels** 를 선택합니다.
4. **Add panels** 메뉴에서 **Scatter plot** 을 선택합니다.
5. 표시하려는 데이터를 `x` 와 `y` 축에 설정합니다. 필요에 따라 축의 최대/최소 범위 또는 `z` 축을 추가할 수 있습니다.
6. **Apply** 를 클릭해 산점도를 만듭니다.
7. 새로 생성된 산점도를 Charts 패널에서 확인할 수 있습니다.