---
title: Scatter plots
menu:
  default:
    identifier: ko-guides-models-app-features-panels-scatter-plot
    parent: panels
weight: 40
---

산점도를 사용하여 여러 run을 비교하고 실험이 어떻게 수행되고 있는지 시각화합니다. 다음과 같은 사용자 정의 가능한 기능이 추가되었습니다.

1. 최소, 최대 및 평균을 따라 선을 플롯합니다.
2. 사용자 정의 메타데이터 툴팁
3. 제어점 색상
4. 축 범위 설정
5. 축을 로그 스케일로 전환

다음은 몇 주간의 실험에서 다양한 model의 검증 정확도에 대한 예입니다. 툴팁은 배치 크기 및 드롭아웃과 축의 값들을 포함하도록 사용자 정의되었습니다. 또한 검증 정확도의 이동 평균을 플로팅하는 선도 있습니다.
[라이브 예제 보기 →](https://app.wandb.ai/l2k2/l2k/reports?view=carey%2FScatter%20Plot)

{{< img src="/images/general/scatter-plots-1.png" alt="" >}}
