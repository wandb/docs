---
description: Visualize the results of your W&B Sweeps with the W&B App UI.
displayed_sidebar: default
---

# 스윕 결과 시각화하기

<head>
  <title>W&B 스윕의 결과 시각화하기</title>
</head>

W&B 스윕의 결과를 W&B 앱 UI를 통해 시각화하세요. [https://wandb.ai/home](https://wandb.ai/home)에서 W&B 앱 UI로 이동하세요. W&B 스윕을 초기화할 때 지정한 프로젝트를 선택하세요. 그러면 프로젝트 [워크스페이스](../app/pages/workspaces.md)로 리디렉션됩니다. 왼쪽 패널에 있는 **스윕 아이콘** (빗자루 아이콘)을 선택하세요. [스윕 UI](./visualize-sweep-results.md)에서 스윕 목록에서 스윕의 이름을 선택하세요.

기본적으로, W&B는 스윕 작업을 시작할 때 자동으로 평행 좌표 플롯, 파라미터 중요도 플롯 및 산점도를 생성합니다.

![스윕 UI 인터페이스로 이동하여 자동 생성된 플롯을 보는 방법을 보여주는 애니메이션.](/images/sweeps/navigation_sweeps_ui.gif)

평행 좌표 차트는 대량의 하이퍼파라미터와 모델 메트릭 간의 관계를 한눈에 요약합니다. 평행 좌표 플롯에 대한 자세한 정보는 [평행 좌표](../app/features/panels/parallel-coordinates.md)를 참조하세요.

![평행 좌표 플롯 예시.](/images/sweeps/example_parallel_coordiantes_plot.png)

산점도(왼쪽)는 스윕 동안 생성된 W&B 실행을 비교합니다. 산점도에 대한 자세한 정보는 [산점도](../app/features/panels/scatter-plot.md)를 참조하세요.

파라미터 중요도 플롯(오른쪽)은 메트릭의 바람직한 값과 높은 상관관계를 가진 최고의 예측자인 하이퍼파라미터를 나열합니다. 파라미터 중요도 플롯에 대한 자세한 정보는 [파라미터 중요도](../app/features/panels/parameter-importance.md)를 참조하세요.

![산점도 예시(왼쪽) 및 파라미터 중요도 플롯 예시(오른쪽).](/images/sweeps/scatter_and_parameter_importance.png)


자동으로 사용되는 독립 변수 및 종속 변수(x축 및 y축)를 변경할 수 있습니다. 각 패널 내에는 **패널 편집**이라고 불리는 연필 아이콘이 있습니다. **패널 편집**을 선택하세요. 모달이 나타납니다. 모달 내에서 그래프의 동작을 변경할 수 있습니다.

모든 기본 W&B 시각화 옵션에 대한 자세한 정보는 [패널](../app/features/panels/intro.md)을 참조하세요. W&B 스윕의 일부가 아닌 W&B 실행에서 플롯을 생성하는 방법에 대한 정보는 [데이터 시각화 문서](../tables/intro.md)를 참조하세요.