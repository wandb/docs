---
title: Visualize sweep results
description: W&B App UI를 사용하여 W&B Sweeps의 결과를 시각화하세요.
menu:
  default:
    identifier: ko-guides-models-sweeps-visualize-sweep-results
    parent: sweeps
weight: 7
---

W&B App UI를 사용하여 W&B Sweeps 의 결과를 시각화합니다. [https://wandb.ai/home](https://wandb.ai/home)에서 W&B App UI로 이동합니다. W&B Sweep을 초기화할 때 지정한 project를 선택합니다. project [workspace]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})로 리디렉션됩니다. 왼쪽 panel에서 **Sweep 아이콘**(빗자루 아이콘)을 선택합니다. [Sweep UI]({{< relref path="./visualize-sweep-results.md" lang="ko" >}})에서 목록에서 Sweep 이름을 선택합니다.

기본적으로 W&B는 W&B Sweep 작업을 시작할 때 평행 좌표 플롯, 파라미터 중요도 플롯 및 산점도를 자동으로 생성합니다.

{{< img src="/images/sweeps/navigation_sweeps_ui.gif" alt="Sweep UI 인터페이스로 이동하여 자동 생성된 플롯을 보는 방법을 보여주는 애니메이션입니다." >}}

평행 좌표 차트는 많은 수의 하이퍼파라미터 와 model metrics 간의 관계를 한눈에 요약합니다. 평행 좌표 플롯에 대한 자세한 내용은 [평행 좌표]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ko" >}})를 참조하십시오.

{{< img src="/images/sweeps/example_parallel_coordiantes_plot.png" alt="평행 좌표 플롯의 예." >}}

산점도(왼쪽)는 Sweep 중에 생성된 W&B Runs 을 비교합니다. 산점도에 대한 자세한 내용은 [산점도]({{< relref path="/guides/models/app/features/panels/scatter-plot.md" lang="ko" >}})를 참조하십시오.

파라미터 중요도 플롯(오른쪽)은 metrics 의 바람직한 value 와 가장 잘 예측하고 높은 상관 관계가 있는 하이퍼파라미터 를 나열합니다. 자세한 내용은 파라미터 중요도 플롯은 [파라미터 중요도]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ko" >}})를 참조하십시오.

{{< img src="/images/sweeps/scatter_and_parameter_importance.png" alt="산점도 (왼쪽) 및 파라미터 중요도 플롯 (오른쪽)의 예." >}}

자동으로 사용되는 종속 및 독립 value (x 및 y 축)를 변경할 수 있습니다. 각 panel 에는 **Edit panel** 이라는 연필 아이콘이 있습니다. **Edit panel** 을 선택합니다. model 이 나타납니다. 모달 내에서 그래프의 행동을 변경할 수 있습니다.

모든 기본 W&B visualization 옵션에 대한 자세한 내용은 [Panels]({{< relref path="/guides/models/app/features/panels/" lang="ko" >}})를 참조하십시오. W&B Sweep의 일부가 아닌 W&B Runs 에서 플롯을 만드는 방법에 대한 자세한 내용은 [Data Visualization docs]({{< relref path="/guides/models/tables/" lang="ko" >}})를 참조하십시오.
