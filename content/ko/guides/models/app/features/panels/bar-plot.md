---
title: 막대 그래프
description: 메트릭을 시각화하고, 축을 커스터마이즈하며, 범주형 데이터를 막대 그래프로 비교하세요.
menu:
  default:
    identifier: ko-guides-models-app-features-panels-bar-plot
    parent: panels
weight: 20
---

막대 그래프는 범주형 데이터를 직사각형 막대로 표현하며, 수직 또는 수평으로 그릴 수 있습니다. 모든 로그 값의 길이가 1일 때 `wandb.Run.log()`를 사용하면 기본적으로 막대 그래프가 표시됩니다.

{{< img src="/images/app_ui/bar_plot.png" alt="W&B에서 Box와 수평 막대 그래프 그리기" >}}

차트 설정을 통해 최대 표시할 run 수를 제한하고, run 을 원하는 config 별로 그룹화하며 라벨명을 변경할 수 있습니다.

{{< img src="/images/app_ui/bar_plot_custom.png" alt="사용자 지정 막대 그래프" >}}

## 막대 그래프 커스터마이즈하기

**Box** 또는 **Violin** 플롯도 만들어 여러 요약 통계값을 한 번에 시각화할 수 있습니다.

1. runs 테이블에서 run 들을 그룹화하세요.
2. 워크스페이스에서 'Add panel'을 클릭하세요.
3. 표준 'Bar Chart'를 추가하고 시각화할 metric 을 선택하세요.
4. 'Grouping' 탭에서 'box plot', 'Violin' 등을 선택해 원하는 스타일로 그릴 수 있습니다.

{{< img src="/images/app_ui/bar_plots.gif" alt="막대 그래프 커스터마이즈" >}}