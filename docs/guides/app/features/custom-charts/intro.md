---
title: Custom charts
slug: /guides/app/features/custom-charts
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

**커스텀 차트**를 사용하여 현재 기본 UI에서 불가능한 차트를 생성할 수 있습니다. 임의의 데이터 테이블을 로그하고 원하는 대로 시각화하세요. [Vega](https://vega.github.io/vega/)의 힘을 이용하여 폰트, 색상, 툴팁의 세부 사항을 제어하세요.

* **가능한 것**: [런칭 공지](https://wandb.ai/wandb/posts/reports/Announcing-the-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)를 읽어보세요.
* **코드**: [호스팅된 노트북](https://tiny.cc/custom-charts)에서 라이브 예제를 시도해 보세요.
* **비디오**: 빠른 [동영상 가이드](https://www.youtube.com/watch?v=3-N9OV6bkSM)를 보세요.
* **예제**: 빠른 Keras 및 Sklearn [데모 노트북](https://colab.research.google.com/drive/1g-gNGokPWM2Qbc8p1Gofud0_5AoZdoSD?usp=sharing)

![vega.github.io/vega에서 지원하는 차트](/images/app_ui/supported_charts.png)

### 작동 방식

1. **데이터 로그**: 스크립트에서 [config](../../../../guides/track/config.md)와 요약 데이터를 W&B에서 일반적으로 하는 방식으로 로그하세요. 특정 시간에 여러 값이 로그된 목록을 시각화하려면 사용자 지정 `wandb.Table`을 사용하세요.
2. **차트 사용자 지정**: [GraphQL](https://graphql.org) 쿼리를 사용하여 이 로그된 데이터를 가져옵니다. 강력한 시각화 문법인 [Vega](https://vega.github.io/vega/)를 사용하여 쿼리 결과를 시각화하세요.
3. **차트 로그**: 스크립트에서 `wandb.plot_table()`을 사용하여 사용자 지정 설정을 호출하세요.

![](/images/app_ui/pr_roc.png)

## 스크립트에서 차트 로그하기

### 내장된 프리셋

이 프리셋은 `wandb.plot` 메서드를 내장하고 있어 스크립트에서 차트를 직접 로그하고 UI에서 원하는 시각화를 정확히 볼 수 있습니다.

<Tabs
  defaultValue="line-plot"
  values={[
    {label: '라인 플롯', value: 'line-plot'},
    {label: '산점도', value: 'scatter-plot'},
    {label: '막대 그래프', value: 'bar-chart'},
    {label: '히스토그램', value: 'histogram'},
    {label: 'PR 곡선', value: 'pr-curve'},
    {label: 'ROC 곡선', value: 'roc-curve'},
  ]}>
  <TabItem value="line-plot">

`wandb.plot.line()`

임의의 축 x와 y에 (x, y) 연결된 순서의 점 목록인 사용자 지정 줄 플롯을 로그합니다.

```python
data = [[x, y] for (x, y) in zip(x_values, y_values)]
table = wandb.Table(data=data, columns=["x", "y"])
wandb.log(
    {
        "my_custom_plot_id": wandb.plot.line(
            table, "x", "y", title="Custom Y vs X Line Plot"
        )
    }
)
```

이 함수를 사용하여 모든 두 차원에서 커브를 로그할 수 있습니다. 두 목록의 값을 서로 비교하여 로그할 경우, 목록의 값 수는 반드시 정확히 일치해야 합니다 (즉, 각 점은 x와 y를 가져야 합니다).

![](/images/app_ui/line_plot.png)

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

[코드 실행하기](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="scatter-plot">

`wandb.plot.scatter()`

임의의 축 x와 y에 (x, y) 점 목록인 사용자 지정 산점도를 로그합니다.

```python
data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
table = wandb.Table(data=data, columns=["class_x", "class_y"])
wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
```

이 함수를 사용하여 모든 두 차원에서 산점도를 로그할 수 있습니다. 두 목록의 값을 서로 비교하여 로그할 경우, 목록의 값 수는 반드시 정확히 일치해야 합니다 (즉, 각 점은 x와 y를 가져야 합니다).

![](/images/app_ui/demo_scatter_plot.png)

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

[코드 실행하기](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="bar-chart">

`wandb.plot.bar()`

레이블이 붙은 값을 막대로 나타낸 사용자 지정 막대 차트를 몇 줄로 네이티브하게 로그합니다:

```python
data = [[label, val] for (label, val) in zip(labels, values)]
table = wandb.Table(data=data, columns=["label", "value"])
wandb.log(
    {
        "my_bar_chart_id": wandb.plot.bar(
            table, "label", "value", title="Custom Bar Chart"
        )
    }
)
```

이 함수를 사용하여 임의의 막대 그래프를 로그할 수 있습니다. 목록의 레이블과 값의 수가 정확히 일치해야 합니다 (즉, 각 데이터 포인트는 둘 다 있어야 합니다).

![](/images/app_ui/line_plot_bar_chart.png)

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

[코드 실행하기](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="histogram">

`wandb.plot.histogram()`

값 목록을 빈(bin) 단위로 정렬하여 빈도가 발생한 횟수로 구성된 사용자 지정 히스토그램을 몇 줄로 네이티브하게 로그합니다. 예를 들어, 예측 확신도 점수(`scores`) 목록이 있고 그 분포를 시각화하고 싶을 때:

```python
data = [[s] for s in scores]
table = wandb.Table(data=data, columns=["scores"])
wandb.log({"my_histogram": wandb.plot.histogram(table, "scores", title=None)})
```

이 함수를 사용하여 임의의 히스토그램을 로그할 수 있습니다. `data`는 2D 행렬의 행과 열을 지원하기 위한 리스트 리스입니다.

![](/images/app_ui/demo_custom_chart_histogram.png)

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

[코드 실행하기](https://tiny.cc/custom-charts)

  </TabItem>
  <TabItem value="pr-curve">

`wandb.plot.pr_curve()`

[Precision-Recall 곡선](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)을 한 줄로 생성합니다:

```python
plot = wandb.plot.pr_curve(ground_truth, predictions, labels=None, classes_to_plot=None)

wandb.log({"pr": plot})
```

다음과 같은 경우에 코드를 사용하여 로그할 수 있습니다:

* 예제의 모델 예측 점수 (`predictions`)
* 해당 예제에 대한 그라운드 트루스 레이블 (`ground_truth`)
* (옵션) 레이블/클래스 이름 목록 (`labels=["cat", "dog", "bird"...]` 레이블 인덱스 0은 cat, 1은 dog, 2는 bird 등임)
* (옵션) 차트에서 시각화할 레이블의 서브셋 (여전히 목록 형식)

![](/images/app_ui/demo_average_precision_lines.png)

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

[코드 실행하기](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)

  </TabItem>
  <TabItem value="roc-curve">

`wandb.plot.roc_curve()`

[ROC 곡선](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)을 한 줄로 생성합니다:

```python
plot = wandb.plot.roc_curve(
    ground_truth, predictions, labels=None, classes_to_plot=None
)

wandb.log({"roc": plot})
```

다음과 같은 경우에 코드를 사용하여 로그할 수 있습니다:

* 예제의 모델 예측 점수 (`predictions`)
* 해당 예제에 대한 그라운드 트루스 레이블 (`ground_truth`)
* (옵션) 레이블/클래스 이름 목록 (`labels=["cat", "dog", "bird"...]` 레이블 인덱스 0은 cat, 1은 dog, 2는 bird 등임)
* (옵션) 차트에서 시각화할 레이블의 서브셋 (여전히 목록 형식)

![](/images/app_ui/demo_custom_chart_roc_curve.png)

[앱에서 보기](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

[코드 실행하기](https://colab.research.google.com/drive/1_RMppCqsA8XInV_jhJz32NCZG6Z5t1RO?usp=sharing)

  </TabItem>
</Tabs>

### 커스텀 프리셋

내장된 프리셋을 수정하거나 새 프리셋을 생성한 후 차트를 저장합니다. 차트 ID를 사용하여 스크립트에서 해당 맞춤 프리셋으로 직접 데이터를 로그합니다.

```python
# 플롯할 열이 있는 테이블 생성
table = wandb.Table(data=data, columns=["step", "height"])

# 테이블의 열을 차트의 필드에 매핑
fields = {"x": "step", "value": "height"}

# 새로운 커스텀 차트 프리셋을 채우기 위해 테이블 사용
# 저장한 차트 프리셋을 사용하려면 vega_spec_name을 변경하세요
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
)
```

[코드 실행하기](https://tiny.cc/custom-charts)

![](/images/app_ui/custom_presets.png)

## 데이터 로그하기

스크립트에서 로그하고 커스텀 차트로 사용할 수 있는 데이터 유형은 다음과 같습니다:

* **Config**: 실험의 초기 설정 (독립 변수). 트레이닝 시작 시 `wandb.config`에 키로 로그한 명명된 필드를 포함합니다 (예: `wandb.config.learning_rate = 0.0001`).
* **Summary**: 트레이닝 중에 로그된 단일 값 (결과 또는 종속 변수), 예: `wandb.log({"val_acc" : 0.8})`. 트레이닝 중에 여러 번 이 키에 기록하면 요약은 해당 키의 최종 값으로 설정됩니다.
* **History**: 로깅된 스칼라의 전체 시계열이 쿼리를 통해 `history` 필드로 제공됩니다.
* **summaryTable**: 여러 값을 로그해야 할 경우 `wandb.Table()`을 사용하여 해당 데이터를 저장한 다음 커스텀 패널에서 쿼리하세요.
* **historyTable**: 기록 데이터를 봐야 하는 경우, 커스텀 차트 패널에서 `historyTable`을 쿼리합니다. `wandb.Table()`이나 사용자 정의 차트를 로그할 때마다 해당 스텝에 대한 새로운 테이블이 기록에 생성됩니다.

### 사용자 정의 테이블 로그하는 방법

`wandb.Table()`을 사용하여 데이터를 2D 배열로 로그하세요. 일반적으로 이 테이블의 각 행은 하나의 데이터 포인트를 나타내며, 각 열은 플롯하려는 각 데이터 포인트의 관련 필드/차원을 나타냅니다. 커스텀 패널을 구성할 때 `wandb.log()`에 전달된 명명된 키("custom_data_table" 아래)를 통해 전체 테이블에 엑세스할 수 있으며, 개별 필드는 열 이름("x", "y", "z")을 통해 엑세스할 수 있습니다. 실험 중 여러 시간 단계에서 테이블을 로그할 수 있습니다. 각 테이블의 최대 크기는 10,000행입니다.

[Google Colab에서 시도해보기](https://tiny.cc/custom-charts)

```python
# 데이터의 사용자 정의 테이블 로그하기
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

## 차트 사용자 정의하기

새 사용자 정의 차트를 추가하여 시작한 다음, 쿼리를 편집하여 보이는 Runs 에서 데이터를 선택합니다. 이 쿼리는 [GraphQL](https://graphql.org)을 사용하여 Runs 의 config, summary 및 history 필드에서 데이터를 가져옵니다.

![새 사용자 정의 차트를 추가한 후 쿼리를 편집하세요](/images/app_ui/customize_chart.gif)

### 사용자 정의 시각화

오른쪽 상단 모서리에서 **차트**를 선택하여 기본 프리셋으로 시작합니다. 그런 다음, 쿼리에서 가져오는 데이터를 차트의 해당 필드에 매핑하기 위해 **차트 필드**를 선택합니다. 쿼리에서 메트릭을 선택한 다음 아래의 막대 차트 필드에 매핑하는 예를 들어보겠습니다.

![프로젝트에서 Runs의 정확도를 보여주는 사용자 정의 막대 차트 생성](/images/app_ui/demo_make_a_custom_chart_bar_chart.gif)

### Vega 편집 방법

패널 상단에서 **편집**을 클릭하여 [Vega](https://vega.github.io/vega/) 편집 모드로 들어갑니다. 여기서 UI에서 대화형 차트를 생성하는 [Vega 규격](https://vega.github.io/vega/docs/specification/)을 정의할 수 있습니다. 차트의 시각적 스타일 (예: 제목 변경, 다른 색상 체계 선택, 시리즈 대신 포인트로 커브 표시)부터 데이터 자체 (Vega 변환을 사용하여 값을 히스토그램으로 빈에 배치)까지 모든 측면을 변경할 수 있습니다. 패널 미리보기는 대화식으로 업데이트되므로 Vega 사양이나 쿼리를 편집할 때 변경 사항의 효과를 확인할 수 있습니다. [Vega 문서 및 튜토리얼](https://vega.github.io/vega/)은 영감을 얻는 훌륭한 자료입니다.

**필드 참조**

W&B에서 차트로 데이터를 가져오려면 Vega 사양의 어느 곳에나 `"${field:<field-name>}"` 형식의 템플릿 문자열을 추가하세요. 이렇게 하면 오른쪽 **차트 필드** 영역에 드롭다운이 생성되어, 사용자가 Vega에 매핑하기 위해 쿼리 결과 열을 선택할 수 있습니다.

필드의 기본값을 설정하려면 이 구문을 사용하세요: `"${field:<field-name>:<placeholder text>}"`

### 차트 프리셋 저장하기

모달 하단의 버튼으로 특정 시각화 패널에 변경 사항을 적용합니다. 또는 다른 프로젝트에서도 Vega 사양을 저장하여 사용할 수 있습니다. 재사용 가능한 차트 정의를 저장하려면 Vega 에디터 상단에서 **다른 이름으로 저장**을 클릭하고 프리셋에 이름을 지정하세요.

## 문서 및 가이드

1. [The W&B Machine Learning Visualization IDE](https://wandb.ai/wandb/posts/reports/The-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)
2. [Visualizing NLP Attention Based Models](https://wandb.ai/kylegoyette/gradientsandtranslation2/reports/Visualizing-NLP-Attention-Based-Models-Using-Custom-Charts--VmlldzoyNjg2MjM)
3. [Visualizing The Effect of Attention on Gradient Flow](https://wandb.ai/kylegoyette/gradientsandtranslation/reports/Visualizing-The-Effect-of-Attention-on-Gradient-Flow-Using-Custom-Charts--VmlldzoyNjg1NDg)
4. [Logging arbitrary curves](https://wandb.ai/stacey/presets/reports/Logging-Arbitrary-Curves--VmlldzoyNzQyMzA)

## 자주 묻는 질문

### 곧 선보입니다

* **폴링**: 차트에서 데이터 자동 새로 고침
* **샘플링**: 효율성을 위해 패널에 로드된 총 포인트 수를 동적으로 조정

### 알아야 할 사항

* 차트를 편집하는 중에 예상했던 데이터를 쿼리에서 보지 못하나요? 선택한 Runs 에서 로그되지 않은 열일 수 있습니다. 차트를 저장한 후 Runs 테이블로 돌아가 **눈** 아이콘으로 시각화할 Runs 를 선택하세요.

### 커스텀 차트에서 "단계 슬라이더"를 표시하는 방법은?

이 기능은 커스텀 차트 편집기의 "기타 설정" 페이지에서 활성화할 수 있습니다. 쿼리를 `summaryTable` 대신 `historyTable`을 사용하도록 변경하면, 커스텀 차트 편집기에서 "단계 선택기 표시" 옵션을 사용할 수 있습니다. 이것은 단계 선택을 가능하게 하는 슬라이더를 제공합니다.

### 커스텀 차트 프리셋을 삭제하는 방법은?


커스텀 차트 편집기로 들어가세요. 현재 선택한 차트 유형을 클릭하면 모든 프리셋이 포함된 메뉴가 열립니다. 삭제하려는 프리셋에 마우스를 올리고 쓰레기통 아이콘을 클릭하세요. 

![](/images/app_ui/delete_custome_chart_preset.gif)

### 일반적인 유스 케이스

* 오차 막대를 사용하여 막대형 플롯 사용자 지정
* PR 곡선과 같이 사용자 지정 x-y 좌표를 요구하는 모델 검증 메트릭 표시
* 두 개의 다른 모델/Experiments에서 데이터를 히스토그램으로 오버레이
* 트레이닝 중 여러 시점에서 스냅샷을 통한 메트릭 변화 표시
* W&B에 아직 제공되지 않은 고유의 시각화 생성 (그리고 이를 세계와 공유하기)