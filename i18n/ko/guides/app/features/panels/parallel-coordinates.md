---
description: Compare results across machine learning experiments
displayed_sidebar: default
---

# 평행 좌표

평행 좌표 차트는 다수의 하이퍼파라미터와 모델 메트릭 간의 관계를 한눈에 요약합니다.

![](/images/app_ui/parallel_coordinates.gif)

* **축**: [`wandb.config`](../../../../guides/track/config.md)에서 가져온 다양한 하이퍼파라미터와 [`wandb.log`](../../../../guides/track/log/intro.md)에서 가져온 메트릭.
* **선**: 각 선은 단일 run을 나타냅니다. 선 위로 마우스를 올리면 run에 대한 자세한 툴팁이 표시됩니다. 현재 필터와 일치하는 모든 선이 표시되지만, 눈 모양을 끄면 선이 회색으로 표시됩니다.

## 패널 설정

패널 설정에서 이러한 기능을 구성하세요— 패널 오른쪽 상단의 편집 버튼을 클릭합니다.

* **툴팁**: 마우스를 올리면 각 run에 대한 정보가 담긴 범례가 표시됩니다
* **제목**: 축 제목을 더 읽기 쉽게 편집합니다
* **그레이디언트**: 원하는 색상 범위로 그레이디언트를 사용자 지정합니다
* **로그 스케일**: 각 축은 독립적으로 로그 스케일로 보기 설정할 수 있습니다
* **축 뒤집기**: 축 방향을 전환합니다— 이는 정확도와 손실을 모두 열로 가지고 있을 때 유용합니다

[실시간으로 보기 →](https://app.wandb.ai/example-team/sweep-demo/reports/Zoom-in-on-Parallel-Coordinates-Charts--Vmlldzo5MTQ4Nw)