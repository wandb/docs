---
title: Visualize sweep results
description: W&B Sweeps의 결과는 W&B App UI에서 시각화할 수 있습니다.
displayed_sidebar: default
---

W&B App UI를 사용하여 W&B Sweeps의 결과를 시각화하세요. [https://wandb.ai/home](https://wandb.ai/home)에서 W&B App UI로 이동합니다. W&B Sweep을 초기화할 때 지정한 프로젝트를 선택합니다. 그러면 프로젝트 [workspace](../app/pages/workspaces.md)로 리디렉션됩니다. 왼쪽 패널에서 **Sweeps 아이콘**(빗자루 아이콘)을 선택합니다. [Sweep UI](./visualize-sweep-results.md)에서 리스트에서 스윕 이름을 선택합니다.

기본적으로, W&B는 W&B Sweep 작업을 시작할 때 자동으로 평행 좌표 플롯, 파라미터 중요도 플롯, 산점도를 생성합니다.

![Sweep UI 인터페이스로 이동하여 자동 생성된 플롯을 보는 방법을 보여주는 애니메이션.](/images/sweeps/navigation_sweeps_ui.gif)

평행 좌표 차트는 많은 하이퍼파라미터와 모델 메트릭 간의 관계를 한 눈에 요약합니다. 평행 좌표 플롯에 대한 자세한 내용은 [Parallel coordinates](../app/features/panels/parallel-coordinates.md)를 참조하세요.

![평행 좌표 플롯 예시.](/images/sweeps/example_parallel_coordiantes_plot.png)

산점도(왼쪽)는 스윕 중에 생성된 W&B Runs를 비교합니다. 산점도에 대한 자세한 내용은 [Scatter Plots](../app/features/panels/scatter-plot.md)를 참조하세요.

파라미터 중요도 플롯(오른쪽)은 메트릭의 바람직한 값과 높게 상관된, 최고의 예측력을 가진 하이퍼파라미터 목록을 보여줍니다. 파라미터 중요도 플롯에 대한 자세한 내용은 [Parameter Importance](../app/features/panels/parameter-importance.md)를 참조하세요.

![산점도(왼쪽) 및 파라미터 중요도 플롯(오른쪽) 예시.](/images/sweeps/scatter_and_parameter_importance.png)

자동으로 사용되는 종속 및 독립 값(x 및 y 축)을 변경할 수 있습니다. 각 패널 내에는 **패널 편집(Edit panel)**이라는 연필 아이콘이 있습니다. **패널 편집**을 선택하세요. 모델이 나타납니다. 이 창 내에서 그래프의 행동을 변경할 수 있습니다.

모든 기본 W&B 시각화 옵션에 대한 자세한 내용은 [Panels](../app/features/panels/intro.md)를 참조하세요. W&B Sweep의 일부가 아닌 W&B Runs로부터 플롯을 생성하는 방법에 대한 정보는 [Data Visualization docs](../tables/intro.md)를 참조하세요.