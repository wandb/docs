---
slug: /guides/app/features/custom-charts
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 사용자 정의 차트

**사용자 정의 차트**를 사용하여 현재 기본 UI에서는 불가능한 차트를 생성하세요. 임의의 데이터 테이블을 로그하고 원하는 대로 정확하게 시각화하세요. [Vega](https://vega.github.io/vega/)의 힘을 빌려 글꼴, 색상, 툴팁의 세부 사항을 제어하세요.

* **가능한 것들**: [출시 공지 읽기 →](https://wandb.ai/wandb/posts/reports/Announcing-the-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)
* **코드**: [호스팅 노트북에서 실시간 예제 시도하기 →](https://tiny.cc/custom-charts)
* **비디오**: 빠른 [소개 비디오 시청하기 →](https://www.youtube.com/watch?v=3-N9OV6bkSM)
* **예시**: Keras와 Sklearn의 빠른 [데모 노트북 →](https://colab.research.google.com/drive/1g-gNGokPWM2Qbc8p1Gofud0_5AoZdoSD?usp=sharing)

![vega.github.io/vega에서 지원하는 차트](/images/app_ui/supported_charts.png)

### 작동 방식

1. **데이터 로그하기**: 스크립트에서 [config](../../../../guides/track/config.md) 및 요약 데이터를 W&B와 함께 실행할 때와 같이 평소처럼 로그합니다. 특정 시간에 로그된 여러 값의 목록을 시각화하려면 사용자 정의 `wandb.Table`을 사용하세요.
2. **차트 사용자 정의하기**: [GraphQL](https://graphql.org) 쿼리로 로그된 모든 데이터를 가져옵니다. 쿼리 결과를 [Vega](https://vega.github.io/vega/)를 사용하여 시각화하세요. Vega는 강력한 시각화 문법입니다.
3. **차트 로그하기**: 스크립트에서 자신만의 프리셋을 `wandb.plot_table()`로 호출하세요.

![](/images/app_ui/pr_roc.png)

## 스크립트에서 차트 로그하기

### 내장 프리셋

이 프리셋들은 내장된 `wandb.plot` 메소드를 가지고 있어 스크립트에서 직접 차트를 로그하고 UI에서 정확한 시각화를 빠르게 볼 수 있게 합니다.

<Tabs
  defaultValue="line-plot"
  values={[
    {label: '선 그래프', value: 'line-plot'},
    {label: '산점도', value: 'scatter-plot'},
    {label: '막대 차트', value: 'bar-chart'},
    {label: '히스토그램', value: 'histogram'},
    {label: 'PR 곡선', value: 'pr-curve'},
    {label: 'ROC 곡선', value: 'roc-curve'},
  ]}>
  <TabItem value="line-plot">

`wandb.plot.line()`

임의의 축 x 및 y에서 연결되고 정렬된 점(x,y) 목록인 사용자 정의 선 그래프를 로그합니다.

```python
data = [[x, y] for (x, y) in zip(x_values, y_values)]
table = wandb.Table(data=data, columns=["x", "y"])
wandb.log(
    {
        "my_custom_plot_id": wandb.plot.line(
            table, "x", "y", title="사용자 정의 Y vs X 선 그래프"
        )
    }
)
```

두 차원에서 곡선을 로그하는 데 이를 사용할 수 있습니다. 두 값 목록을 서로 대비하여 플로팅하는 경우, 목록의 값 수가 정확히 일치해야 합니다(즉, 각 점은 x와 y를 가져야 합니다).

![](/images/app_ui/line_plot.png)

[앱에서 보기 →](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

[코드 실행하기 →](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="scatter-plot">

`wandb.plot.scatter()`

임의의 축 x 및 y에서 점(x, y) 목록인 사용자 정의 산점도를 로그합니다.

```python
data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
table = wandb.Table(data=data, columns=["class_x", "class_y"])
wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
```

두 차원에서 산점도를 로그하는 데 이를 사용할 수 있습니다. 두 값 목록을 서로 대비하여 플로팅하는 경우, 목록의 값 수가 정확히 일치해야 합니다(즉, 각 점은 x와 y를 가져야 합니다).

![](/images/app_ui/demo_scatter_plot.png)

[앱에서 보기 →](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

[코드 실행하기 →](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="bar-chart">

`wandb.plot.bar()`

막대로 표시된 레이블이 있는 값 목록인 사용자 정의 막대 차트를 몇 줄로 기본적으로 로그합니다.

```python
data = [[label, val] for (label, val) in zip(labels, values)]
table = wandb.Table(data=data, columns=["label", "value"])
wandb.log(
    {
        "my_bar_chart_id": wandb.plot.bar(
            table, "label", "value", title="사용자 정의 막대 차트"
        )
    }
)
```

임의의 막대 차트를 로그하는 데 이를 사용할 수 있습니다. 목록의 레이블과 값의 수가 정확히 일치해야 합니다(즉, 각 데이터 포인트는 둘 다 있어야 합니다).

![](@site/static/images/app_ui/line_plot_bar_chart.png)

[앱에서 보기 →](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

[코드 실행하기 →](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="histogram">

`wandb.plot.histogram()`

값 목록을 발생 빈도/빈도수로 구분하여 분류하는 사용자 정의 히스토그램을 몇 줄로 기본적으로 로그합니다. 예를 들어, 예측 신뢰도 점수(`scores`) 목록이 있고 그 분포를 시각화하고 싶다고 가정해 보겠습니다:

```python
data = [[s] for s in scores]
table = wandb.Table(data=data, columns=["scores"])
wandb.log({"my_histogram": wandb.plot.histogram(table, "scores", title=None)})
```

임의의 히스토그램을 로그하는 데 이를 사용할 수 있습니다. `data`는 행과 열의 2D 배열을 지원하기 위해 목록의 목록으로 되어 있음에 유의하세요.

![](/images/app_ui/demo_custom_chart_histogram.png)

[앱에서 보기 →](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

[코드 실행하기 →](https://tiny.cc/custom-charts)

  </TabItem>
    <TabItem value="pr-curve">

`wandb.plot.pr_curve()`

한 줄로 [정밀도-재현율 곡선](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)을 생성합니다:

```python
plot = wandb.plot.pr_curve(ground_truth, predictions, labels=None, classes_to_plot=None)

wandb.log({"pr": plot})
```

코드가 다음에 엑세스할 수 있을 때 이를 로그할 수 있습니다:

* 예제 집합에 대한 모델의 예측 점수(`predictions`)
* 해당 예제에 대한 해당 그라운드 트루스 레이블(`ground_truth`)
* (선택적으로) 레이블/클래스 이름 목록(`labels=["cat", "dog", "bird"...]`이면 레이블 인덱스 0은 고양이, 1 = 개, 2 = 새 등을 의미합니다.)
* (선택적으로) 플롯에 시각화할 레이블의 서브셋(여전히 목록 형식)

![](/images/app_ui/demo_average_precision_lines.png)


[앱에서 보기 →](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

[코드 실행하기 →](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)

  </TabItem>
  <TabItem value="roc-curve">

`wandb.plot.roc_curve()`

한 줄로 [ROC 곡선](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)을 생성합니다:

```python
plot = wandb.plot.roc_curve(
    ground_truth, predictions, labels=None, classes_to_plot=None
)

wandb.log({"roc": plot})
```

코드가 다음에 엑세스할 수 있을 때 이를 로그할 수 있습니다:

* 예제 집합에 대한 모델의 예측 점수(`predictions`)
* 해당 예제에 대한 해당 그라운드 트루스 레이블(`ground_truth`)
* (선택적으로) 레이블/클래스 이름 목록(`labels=["cat", "dog", "bird"...]`이면 레이블 인덱스 0은 고양이, 1 = 개, 2 = 새 등을 의미합니다.)
* (선택적으로) 플롯에 시각화할 레이블의 서브셋(여전히 목록 형식)

![](/images/app_ui/demo_custom_chart_roc_curve.png)

[앱에서 보기 →](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

[코드 실행하기 →](https://colab.research.google.com/drive/1_RMppCqsA8XInV_jhJz32NCZG6Z5t1RO?usp=sharing)

  </TabItem>
</Tabs>

### 사용자 정의 프리셋

기본 프리셋을 수정하거나 새 프리셋을 생성한 다음 차트를 저장하세요. 스크립트에서 직접 사용자 정의 프리셋에 데이터를 로그하려면 차트 ID를 사용하세요.

```python
# 플롯할 열이 있는 테이블 생성
table = wandb.Table(data=data, columns=["step", "height"])

# 테이블의 열을 차트의 필드에 매핑
fields = {"x": "step", "value": "height"}

# 테이블을 사용하여 새 사용자 정의 차트 프리셋 채우기
# 자신의 저장된 차트 프리셋을 사용하려면 vega_spec_name 변경
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
)
```

[코드 실행하기 →](https://tiny.cc/custom-charts)

![](/images/app_ui/custom_presets.png)

## 데이터 로그하기

스크립트에서 로그할 수 있는 데이터 유형은 다음과 같습니다:

* **Config**: 실험의 초기 설정(독립 변수). 이는 트레이닝 시작 시 `wandb.config`에 로그된 키로 명명된 모든 필드를 포함합니다(예: `wandb.config.learning_rate = 0.0001`)
* **Summary**: 트레이닝 중에 로그된 단일 값(결과 또는 종속 변수), 예를 들어 `wandb.log({"val_acc" : 0.8})`. `wandb.log()`를 통해 트레이닝 중에 이 키에 여러 번 작성하는 경우, 요약은 해당 키의 최종 값으로 설정됩니다.
* **History**: 로그된 스칼라의 전체 시계열은 `history` 필드를 통해 쿼리 가능
* **summaryTable**: 여러 값을 로그할 필요가 있는 경우 `wandb.Table()`을 사용하여 해당 데이터를 저장한 다음 사용자 정의 패널에서 쿼리하세요.
* **historyTable**: 히스토리 데이터를 보려면 사용자 정의 차트 패널에서 `historyTable`을 쿼리하세요. `wandb.Table()`을 호출하거나 사용자 정의 차트를 로그할 때마다 해당 단계의 히스토리에 새 테이블을 생성합니다.

### 사용자 정의 테이블 로그하는 방법

`wandb.Table()`을 사용하여 데이터를 2D 배열로 로그하세요. 일반적으로 이 테이블의 각 행은 하나의 데이터 포인트를 나타내며, 각 열은 각 데이터 포인트에 대해 플로팅하려는 관련 필드/차원을 나타냅니다. 사용자 정의 패널을 구성할 때, `wandb.log()`에 전달된 명명된 키("custom_data_table" 아래)를 통해 전체 테이블에 엑세스할 수 있으며, 개별 필드는 열 이름("x", "y", "z")을 통해 엑세스할 수 있습니다. 실험 중 여러 시간 단계에서 테이블을 로그할 수 있습니다. 각 테이블의 최대 크기는 10,000행입니다.

[Google Colab에서 시도해보세요 →](https://tiny.cc/custom-charts)

```python
# 데이터의 사용자 정의 테이블 로그하기
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

## 차트 사용자 정의하기

시작하려면 새 사용자 정의 차트를 추가한 다음 쿼리를 편집하여 보이는 실행에서 데이터를 선택하세요. 쿼리는 [GraphQL](https://graphql.org)을 사용하여 실행의 config, summary 및 history 필드에서 데이터를 가져옵니다.

![새 사용자 정의 차트 추가 후 쿼리 편집](/images/app_ui/customize_chart.gif)

### 사용자 정의 시각화

오른쪽 상단에서 **차트**를 선택하여 기본 프리셋으로 시작하세요. 다음으로, 쿼리에서 가져온 데이터를 차트의 해당 필드에 매핑하는 **차트 필드**를 선택하세요. 여기에는 쿼리에서 가져온 메트릭을 선택한 다음 아래의 막대 차트 필드에 매핑하는 예가 있습니다.

![프로젝트에서 실행 간 정확도를 보여주는 사용자 정의 막대 차트 생성](/images/app_ui/demo_make_a_custom_chart_bar_chart.gif)

### Vega 편집 방법

패널 상단에서 **편집**을 클릭하여 [Vega](https://vega.github.io/vega/) 편집 모드로 들어