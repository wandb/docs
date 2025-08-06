---
title: 병렬 좌표
description: 기계학습 실험 간 결과 비교하기
menu:
  default:
    identifier: ko-guides-models-app-features-panels-parallel-coordinates
    parent: panels
weight: 30
---

Parallel coordinates 차트는 많은 수의 하이퍼파라미터와 모델 메트릭 간의 관계를 한눈에 요약해 보여 줍니다.

{{< img src="/images/app_ui/parallel_coordinates.gif" alt="Parallel coordinates plot" >}}

* **축(Axes)**: 다양한 하이퍼파라미터는 [`wandb.Run.config`]({{< relref path="/guides/models/track/config.md" lang="ko" >}})에서, 메트릭은 [`wandb.Run.log()`]({{< relref path="/guides/models/track/log/" lang="ko" >}})에서 가져옵니다.
* **선(Lines)**: 각 선은 하나의 run을 나타냅니다. 선 위에 마우스를 올리면 해당 run에 대한 자세한 내용을 툴팁으로 볼 수 있습니다. 현재 필터와 일치하는 모든 run의 선이 표시되며, 눈 아이콘을 끄면 선이 흐리게 표시됩니다.

## Parallel coordinates 패널 생성하기

1. 자신의 workspace 랜딩 페이지로 이동합니다.
2. **Add Panels**를 클릭합니다.
3. **Parallel coordinates**를 선택합니다.

## 패널 설정

패널을 구성하려면 패널 오른쪽 상단의 편집 버튼을 클릭하세요.

* **Tooltip**: 마우스를 올리면 각 run에 대한 정보가 담긴 범례가 나타납니다.
* **Titles**: 축 제목을 더 읽기 쉽게 수정할 수 있습니다.
* **Gradient**: 원하는 색상 범위로 그레이디언트를 커스터마이즈할 수 있습니다.
* **Log scale**: 각 축을 독립적으로 로그 스케일로 표시할 수 있습니다.
* **Flip axis**: 축의 방향을 전환할 수 있습니다. (정확도와 손실 모두를 컬럼으로 사용할 때 유용합니다.)

[Parallel coordinates 패널을 라이브로 직접 확인해 보세요](https://app.wandb.ai/example-team/sweep-demo/reports/Zoom-in-on-Parallel-Coordinates-Charts--Vmlldzo5MTQ4Nw)