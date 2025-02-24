---
title: Parallel coordinates
description: 기계 학습 Experiments 전반에서 결과를 비교하세요.
menu:
  default:
    identifier: ko-guides-models-app-features-panels-parallel-coordinates
    parent: panels
weight: 30
---

Parallel coordinates 차트는 많은 수의 하이퍼파라미터와 모델 메트릭 간의 관계를 한눈에 요약합니다.

{{< img src="/images/app_ui/parallel_coordinates.gif" alt="" >}}

* **Axes**: [`wandb.config`]({{< relref path="/guides/models/track/config.md" lang="ko" >}})의 다양한 하이퍼파라미터 및 [`wandb.log`]({{< relref path="/guides/models/track/log/" lang="ko" >}})의 메트릭.
* **Lines**: 각 선은 단일 run을 나타냅니다. 선 위에 마우스를 올리면 run에 대한 세부 정보가 담긴 툴팁이 표시됩니다. 현재 필터와 일치하는 모든 선이 표시되지만, 눈 모양 아이콘을 끄면 선이 회색으로 표시됩니다.

## 패널 설정

패널 설정에서 이러한 기능을 구성합니다. 패널 오른쪽 상단 모서리에 있는 편집 버튼을 클릭하세요.

* **Tooltip**: 마우스를 올리면 각 run에 대한 정보가 담긴 범례가 나타납니다.
* **Titles**: 축 제목을 더 읽기 쉽도록 편집합니다.
* **Gradient**: 원하는 색상 범위로 그래디언트를 사용자 정의합니다.
* **Log scale**: 각 축을 독립적으로 로그 스케일로 보도록 설정할 수 있습니다.
* **Flip axis**: 축 방향을 전환합니다. 이는 정확도와 손실을 모두 열로 사용할 때 유용합니다.

[실제로 보기 →](https://app.wandb.ai/example-team/sweep-demo/reports/Zoom-in-on-Parallel-Coordinates-Charts--Vmlldzo5MTQ4Nw)
