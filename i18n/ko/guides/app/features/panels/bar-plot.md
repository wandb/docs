---
description: Visualize metrics, customize axes, and compare categorical data as bars.
displayed_sidebar: default
---

# 막대 그래프

막대 그래프는 수평 또는 수직으로 그려진 직사각형 막대로 범주형 데이터를 표시합니다. 모든 로그된 값이 길이가 1일 때 기본적으로 **wandb.log()**를 사용하여 막대 그래프가 표시됩니다.

![W&B에서 박스 및 수평 막대 그래프 표시](/images/app_ui/bar_plot.png)

차트 설정을 사용자 지정하여 표시할 최대 실행 수를 제한하고, 모든 설정으로 실행을 그룹화하고, 레이블 이름을 변경할 수 있습니다.

![](/images/app_ui/bar_plot_custom.png)

### 막대 그래프 사용자 지정

**Box** 또는 **Violin** 그래프를 생성하여 여러 요약 통계를 하나의 차트 유형으로 결합할 수도 있습니다**.**

1. 실행 테이블을 통해 실행을 그룹화합니다.
2. 워크스페이스에서 '패널 추가'를 클릭합니다.
3. 표준 '막대 차트'를 추가하고 그릴 메트릭을 선택합니다.
4. '그룹화' 탭에서 '박스 플롯' 또는 'Violin' 등을 선택하여 이 스타일 중 하나를 그립니다.

![막대 그래프 사용자 지정](@site/static/images/app_ui/bar_plots.gif)