---
description: Visualize metrics, customize axes, and compare categorical data as bars.
displayed_sidebar: default
---

# 막대 그래프

막대 그래프는 수평 또는 수직으로 그려진 직사각형 막대를 사용하여 범주형 데이터를 표시합니다. 로그된 모든 값의 길이가 하나일 때 **wandb.log()**를 사용하면 기본적으로 막대 그래프가 표시됩니다.

![W&B에서 상자 그림 및 수평 막대 그래프 그리기](/images/app_ui/bar_plot.png)

차트 설정을 사용자 정의하여 표시할 최대 run 수를 제한하고, 모든 설정으로 run을 그룹화하고 레이블 이름을 변경합니다.

![](/images/app_ui/bar_plot_custom.png)

### 막대 그래프 사용자 정의하기

여러 요약 통계를 하나의 차트 유형으로 결합하는 **Box** 또는 **Violin** 그림도 만들 수 있습니다**.**

1. run 테이블을 통해 run을 그룹화합니다.
2. 워크스페이스에서 '패널 추가'를 클릭합니다.
3. 표준 '막대 차트'를 추가하고 그래프에 표시할 메트릭을 선택합니다.
4. '그룹화' 탭에서 'box plot' 또는 'Violin' 등을 선택하여 이러한 스타일 중 하나를 그립니다.

![막대 그래프 사용자 정의하기](@site/static/images/app_ui/bar_plots.gif)