---
title: 스윕 결과 시각화
description: W&B App UI에서 W&B Sweeps의 결과를 시각화하세요.
menu:
  default:
    identifier: ko-guides-models-sweeps-visualize-sweep-results
    parent: sweeps
weight: 7
---

W&B Sweeps 결과를 W&B App에서 시각화할 수 있습니다. [W&B App](https://wandb.ai/home)으로 이동하세요. Sweep을 초기화할 때 지정한 Project 를 선택합니다. Project [workspace]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})로 이동하게 됩니다. 왼쪽 패널에서 **Sweep 아이콘**(빗자루 아이콘)을 선택하세요. Sweep UI에서 목록에서 원하는 Sweep 이름을 선택합니다.

기본적으로, W&B에서는 Sweep job을 시작하면 평행 좌표 플롯, 파라미터 중요도 플롯, 산점도를 자동으로 생성합니다.

{{< img src="/images/sweeps/navigation_sweeps_ui.gif" alt="Sweep UI navigation" >}}

평행 좌표 차트는 많은 하이퍼파라미터와 모델 메트릭의 관계를 한눈에 요약해줍니다. 평행 좌표 플롯에 대한 자세한 내용은 [Parallel coordinates]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ko" >}})를 참고하세요.

{{< img src="/images/sweeps/example_parallel_coordiantes_plot.png" alt="Example parallel coordinates plot." >}}

산점도(왼쪽)는 Sweep 실행 중 생성된 W&B Runs 를 비교합니다. 산점도에 대한 자세한 내용은 [Scatter Plots]({{< relref path="/guides/models/app/features/panels/scatter-plot.md" lang="ko" >}})를 참고하세요.

파라미터 중요도 플롯(오른쪽)은 메트릭의 바람직한 값과 가장 높은 상관관계를 보여주는 최고의 하이퍼파라미터를 나열합니다. 파라미터 중요도 플롯에 대한 자세한 설명은 [Parameter Importance]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ko" >}}) 항목을 참고하세요.

{{< img src="/images/sweeps/scatter_and_parameter_importance.png" alt="Scatter plot and parameter importance" >}}

자동으로 지정되는 종속 및 독립 값(x축과 y축)을 변경할 수 있습니다. 각 패널에서 연필 아이콘인 **Edit panel**을 사용할 수 있습니다. **Edit panel**을 선택하면 모달 창이 나타납니다. 여기서 그래프의 행동을 자유롭게 수정할 수 있습니다.

W&B 기본 시각화 옵션 전체에 대한 자세한 내용은 [Panels]({{< relref path="/guides/models/app/features/panels/" lang="ko" >}})를 참고하세요. W&B Sweep과 별도로 W&B Runs 를 활용해 플롯을 생성하는 방법은 [Data Visualization docs]({{< relref path="/guides/models/tables/" lang="ko" >}})를 참고하세요.