---
title: Bar plots
description: 메트릭을 시각화하고, 축을 사용자 정의하고, 범주형 데이터를 막대로 비교하세요.
menu:
  default:
    identifier: ko-guides-models-app-features-panels-bar-plot
    parent: panels
weight: 20
---

막대 그래프는 범주형 데이터를 직사각형 막대로 나타내며, 이 막대는 수직 또는 수평으로 플롯할 수 있습니다. 모든 기록된 값이 길이가 1인 경우 막대 그래프는 기본적으로 **wandb.log()** 와 함께 표시됩니다.

{{< img src="/images/app_ui/bar_plot.png" alt="Plotting Box and horizontal Bar plots in W&B" >}}

차트 설정을 사용하여 표시할 최대 Runs 수를 제한하고, 모든 config별로 Runs를 그룹화하고, 레이블 이름을 바꿀 수 있습니다.

{{< img src="/images/app_ui/bar_plot_custom.png" alt="" >}}

## 막대 그래프 사용자 정의

**Box** 또는 **Violin** 플롯을 생성하여 여러 요약 통계를 하나의 차트 유형으로 결합할 수도 있습니다.

1. Runs 테이블을 통해 Runs를 그룹화합니다.
2. 워크스페이스에서 '패널 추가'를 클릭합니다.
3. 표준 '막대 차트'를 추가하고 플롯할 메트릭을 선택합니다.
4. '그룹화' 탭에서 'box plot' 또는 'Violin' 등을 선택하여 이러한 스타일 중 하나를 플롯합니다.

{{< img src="/images/app_ui/bar_plots.gif" alt="Customize Bar Plots" >}}