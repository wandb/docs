---
title: Parallel coordinates
description: 기계학습 실험 간의 결과 비교
displayed_sidebar: default
---

대규모의 하이퍼파라미터와 모델 메트릭 간의 관계를 한눈에 요약하는 평행 좌표 차트입니다.

![](/images/app_ui/parallel_coordinates.gif)

* **Axes**: [`wandb.config`](../../../../guides/track/config.md)에서 가져온 다양한 하이퍼파라미터와 [`wandb.log`](../../../../guides/track/log/intro.md)에서 가져온 메트릭.
* **Lines**: 각 선은 하나의 run을 나타냅니다. 선 위에 마우스를 올리면 run에 대한 세부 정보를 포함한 툴팁이 표시됩니다. 현재 필터와 일치하는 모든 선이 표시되지만, 눈 아이콘을 끄면 선들이 회색으로 변합니다.

## 패널 설정

패널 설정에서 이러한 기능을 구성하세요 — 패널의 오른쪽 상단 모서리에 있는 편집 버튼을 클릭하세요.

* **Tooltip**: 마우스를 올리면 각 run에 대한 정보가 포함된 범례가 표시됩니다.
* **Titles**: 축 제목을 더 읽기 쉽게 편집합니다.
* **Gradient**: 원하는 색상 범위로 그레이디언트를 사용자 정의합니다.
* **Log scale**: 각 축을 로그 스케일로 독립적으로 설정할 수 있습니다.
* **Flip axis**: 축 방향을 전환합니다 — 이는 정확도와 손실을 모두 열로 가진 경우에 유용합니다.

[실제로 보기 →](https://app.wandb.ai/example-team/sweep-demo/reports/Zoom-in-on-Parallel-Coordinates-Charts--Vmlldzo5MTQ4Nw)