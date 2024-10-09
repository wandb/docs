---
title: Bar plots
description: 메트릭을 시각화하고, 축을 사용자 정의하며, 범주형 데이터를 막대로 비교합니다.
displayed_sidebar: default
---

막대 그래프는 직사각형 막대를 사용하여 범주형 데이터를 나타내며, 수직 또는 수평으로 그릴 수 있습니다. 모든 로그 값이 길이가 1일 때 **wandb.log()**를 사용하면 기본적으로 막대 그래프가 표시됩니다.

![W&B에서 Box 및 수평 막대 그래프 그리기](/images/app_ui/bar_plot.png)

차트 설정을 통해 최대 run 수를 제한하고, run을 구성별로 그룹화하며 레이블을 변경할 수 있습니다.

![](/images/app_ui/bar_plot_custom.png)

### 막대 그래프 커스터마이즈하기

다양한 요약 통계를 하나의 차트 유형으로 결합하기 위해 **Box** 또는 **Violin** 플롯을 생성할 수도 있습니다.

1. run 테이블을 통해 run을 그룹화합니다.
2. 워크스페이스에서 '패널 추가'를 클릭합니다.
3. 표준 '막대 차트'를 추가하고 그래프를 그릴 지표를 선택합니다.
4. '그룹화' 탭 아래에서 'box plot' 또는 'Violin' 등 원하는 스타일을 선택하여 해당 스타일로 플롯합니다.

![Customize Bar Plots](/images/app_ui/bar_plots.gif)