---
title: Custom charts
cascade:
- url: guides/app/features/custom-charts/:filename
menu:
  default:
    identifier: ko-guides-models-app-features-custom-charts-_index
    parent: w-b-app-ui-reference
url: guides/app/features/custom-charts
weight: 2
---

**Custom Charts**를 사용하여 기본 UI에서 현재 불가능한 차트를 만드세요. 임의의 데이터 테이블을 기록하고 원하는 방식으로 정확하게 시각화하세요. [Vega](https://vega.github.io/vega/)의 강력한 기능을 통해 글꼴, 색상 및 툴팁의 세부 사항을 제어하세요.

* **가능한 것**: [출시 발표](https://wandb.ai/wandb/posts/reports/Announcing-the-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)를 읽어보세요.
* **코드**: [호스팅된 노트북](https://tiny.cc/custom-charts)에서 라이브 예제를 사용해보세요.
* **비디오**: 빠른 [단계별 비디오](https://www.youtube.com/watch?v=3-N9OV6bkSM)를 시청하세요.
* **예제**: 빠른 Keras 및 Sklearn [데모 노트북](https://colab.research.google.com/drive/1g-gNGokPWM2Qbc8p1Gofud0_5AoZdoSD?usp=sharing)

{{< img src="/images/app_ui/supported_charts.png" alt="vega.github.io/vega에서 지원되는 차트" max-width="90%" >}}

### 작동 방식

1. **데이터 기록**: 스크립트에서 W&B로 실행할 때 평소와 같이 [config]({{< relref path="/guides/models/track/config.md" lang="ko" >}}) 및 요약 데이터를 기록합니다. 특정 시점에 기록된 여러 값의 목록을 시각화하려면 사용자 지정 `wandb.Table`을 사용하세요.
2. **차트 사용자 정의**: [GraphQL](https://graphql.org) 쿼리를 사용하여 기록된 이 데이터를 가져옵니다. 강력한 시각화 문법인 [Vega](https://vega.github.io/vega/)를 사용하여 쿼리 결과를 시각화합니다.
3. **차트 기록**: `wandb.plot_table()`로 스크립트에서 자신의 사전 설정을 호출합니다.

{{< img src="/images/app_ui/pr_roc.png" alt="" >}}

## 스크립트에서 차트 기록

### 기본 제공 사전 설정

이러한 사전 설정에는 스크립트에서 직접 차트를 빠르게 기록하고 UI에서 찾고 있는 정확한 시각화를 볼 수 있도록 하는 기본 제공 `wandb.plot` 방법이 있습니다.

{{< tabpane text=true >}}
{{% tab header="선 그래프" value="line-plot" %}}

  `wandb.plot.line()`

  임의의 축 x 및 y에서 연결되고 정렬된 점 목록 (x,y)인 사용자 지정 선 그래프를 기록합니다.

  ```python
  data = [[x, y] for (x, y) in zip(x_values, y_values)]
  table = wandb.Table(data=data, columns=["x", "y"])
  wandb.log(
      {
          "my_custom_plot_id": wandb.plot.line(
              table, "x", "y", title="사용자 지정 Y vs X 선 그래프"
          )
      }
  )
  ```

  이를 사용하여 임의의 두 차원에 대한 곡선을 기록할 수 있습니다. 두 값 목록을 서로 플로팅하는 경우 목록의 값 수가 정확히 일치해야 합니다 (예: 각 점에 x와 y가 있어야 함).

  {{< img src="/images/app_ui/line_plot.png" alt="" >}}

  [앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA)

  [코드 실행](https://tiny.cc/custom-charts)

{{% /tab %}}

{{% tab header="산점도" value="scatter-plot" %}}

  `wandb.plot.scatter()`

  임의의 축 x 및 y 쌍의 점 목록 (x, y)인 사용자 지정 산점도를 기록합니다.

  ```python
  data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
  table = wandb.Table(data=data, columns=["class_x", "class_y"])
  wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
  ```

  이를 사용하여 임의의 두 차원에 대한 산점도를 기록할 수 있습니다. 두 값 목록을 서로 플로팅하는 경우 목록의 값 수가 정확히 일치해야 합니다 (예: 각 점에 x와 y가 있어야 함).

  {{< img src="/images/app_ui/demo_scatter_plot.png" alt="" >}}

  [앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ)

  [코드 실행](https://tiny.cc/custom-charts)

{{% /tab %}}

{{% tab header="막대 차트" value="bar-chart" %}}

  `wandb.plot.bar()`

  몇 줄 안에 기본적으로 레이블이 지정된 값 목록을 막대로 표시하는 사용자 지정 막대 차트를 기록합니다.

  ```python
  data = [[label, val] for (label, val) in zip(labels, values)]
  table = wandb.Table(data=data, columns=["label", "value"])
  wandb.log(
      {
          "my_bar_chart_id": wandb.plot.bar(
              table, "label", "value", title="사용자 지정 막대 차트"
          )
      }
  )
  ```

  이를 사용하여 임의의 막대 차트를 기록할 수 있습니다. 목록의 레이블 및 값 수는 정확히 일치해야 합니다 (예: 각 데이터 포인트에 둘 다 있어야 함).

  {{< img src="/images/app_ui/line_plot_bar_chart.png" alt="" >}}

  [앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk)

  [코드 실행](https://tiny.cc/custom-charts)
{{% /tab %}}

{{% tab header="히스토그램" value="histogram" %}}

  `wandb.plot.histogram()`

  몇 줄 안에 기본적으로 값 목록을 발생 횟수/빈도별로 bin으로 정렬하는 사용자 지정 히스토그램을 기록합니다. 예측 신뢰도 점수 목록 (`scores`)이 있고 해당 분포를 시각화한다고 가정해 보겠습니다.

  ```python
  data = [[s] for s in scores]
  table = wandb.Table(data=data, columns=["scores"])
  wandb.log({"my_histogram": wandb.plot.histogram(table, "scores", title=None)})
  ```

  이를 사용하여 임의의 히스토그램을 기록할 수 있습니다. `data`는 행과 열의 2D 배열을 지원하기 위한 것으로 목록의 목록입니다.

  {{< img src="/images/app_ui/demo_custom_chart_histogram.png" alt="" >}}

  [앱에서 보기](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM)

  [코드 실행](https://tiny.cc/custom-charts)

{{% /tab %}}

{{% tab header="PR 곡선" value="pr-curve" %}}

  `wandb.plot.pr_curve()`

  한 줄로 [Precision-Recall 곡선](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)을 만듭니다.

  ```python
  plot = wandb.plot.pr_curve(ground_truth, predictions, labels=None, classes_to_plot=None)

  wandb.log({"pr": plot})
  ```

  다음 정보에 액세스할 수 있는 경우 언제든지 이를 기록할 수 있습니다.

  * 예제 집합에 대한 모델의 예측 점수 (`predictions`)
  * 해당 예제에 대한 해당 ground truth 레이블 (`ground_truth`)
  * (선택 사항) 레이블/클래스 이름 목록 (`labels=["cat", "dog", "bird"...]` (레이블 인덱스 0이 cat, 1 = dog, 2 = bird 등을 의미하는 경우))
  * (선택 사항) 플롯에서 시각화할 레이블의 서브셋 (여전히 목록 형식)

  {{< img src="/images/app_ui/demo_average_precision_lines.png" alt="" >}}


  [앱에서 보기](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)

  [코드 실행](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing)

{{% /tab %}}

{{% tab header="ROC 곡선" value="roc-curve" %}}

  `wandb.plot.roc_curve()`

  한 줄로 [ROC 곡선](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)을 만듭니다.

  ```python
  plot = wandb.plot.roc_curve(
      ground_truth, predictions, labels=None, classes_to_plot=None
  )

  wandb.log({"roc": plot})
  ```

  다음 정보에 액세스할 수 있는 경우 언제든지 이를 기록할 수 있습니다.

  * 예제 집합에 대한 모델의 예측 점수 (`predictions`)
  * 해당 예제에 대한 해당 ground truth 레이블 (`ground_truth`)
  * (선택 사항) 레이블/클래스 이름 목록 (`labels=["cat", "dog", "bird"...]` (레이블 인덱스 0이 cat, 1 = dog, 2 = bird 등을 의미하는 경우))
  * (선택 사항) 플롯에서 시각화할 이러한 레이블의 서브셋 (여전히 목록 형식)

  {{< img src="/images/app_ui/demo_custom_chart_roc_curve.png" alt="" >}}

  [앱에서 보기](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)

  [코드 실행](https://colab.research.google.com/drive/1_RMppCqsA8XInV_jhJz32NCZG6Z5t1RO?usp=sharing)

{{% /tab %}}
{{< /tabpane >}}

### 사용자 지정 사전 설정

기본 제공 사전 설정을 조정하거나 새 사전 설정을 만든 다음 차트를 저장합니다. 차트 ID를 사용하여 스크립트에서 직접 해당 사용자 지정 사전 설정에 데이터를 기록합니다.

```python
# 플로팅할 열이 있는 테이블을 만듭니다.
table = wandb.Table(data=data, columns=["step", "height"])

# 테이블의 열에서 차트의 필드로 매핑합니다.
fields = {"x": "step", "value": "height"}

# 테이블을 사용하여 새 사용자 지정 차트 사전 설정을 채웁니다.
# 자신의 저장된 차트 사전 설정을 사용하려면 vega_spec_name을 변경하십시오.
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
)
```

[코드 실행](https://tiny.cc/custom-charts)

{{< img src="/images/app_ui/custom_presets.png" alt="" max-width="90%" >}}

## 데이터 기록

다음은 스크립트에서 기록하고 사용자 지정 차트에서 사용할 수 있는 데이터 유형입니다.

* **Config**: experiment의 초기 설정 (독립 변수). 여기에는 트레이닝 시작 시 `wandb.config`에 키로 기록한 명명된 필드가 포함됩니다. 예: `wandb.config.learning_rate = 0.0001`
* **Summary**: 트레이닝 중에 기록된 단일 값 (결과 또는 종속 변수). 예: `wandb.log({"val_acc" : 0.8})`. `wandb.log()`를 통해 트레이닝 중에 이 키에 여러 번 쓰는 경우 요약은 해당 키의 최종 값으로 설정됩니다.
* **History**: 기록된 스칼라의 전체 시계열은 `history` 필드를 통해 쿼리에 사용할 수 있습니다.
* **summaryTable**: 여러 값 목록을 기록해야 하는 경우 `wandb.Table()`을 사용하여 해당 데이터를 저장한 다음 사용자 지정 패널에서 쿼리합니다.
* **historyTable**: 기록 데이터를 봐야 하는 경우 사용자 지정 차트 패널에서 `historyTable`을 쿼리합니다. `wandb.Table()`을 호출하거나 사용자 지정 차트를 기록할 때마다 해당 단계에 대한 history에 새 테이블을 만듭니다.

### 사용자 지정 테이블을 기록하는 방법

`wandb.Table()`을 사용하여 데이터를 2D 배열로 기록합니다. 일반적으로 이 테이블의 각 행은 하나의 데이터 포인트를 나타내고 각 열은 플로팅하려는 각 데이터 포인트에 대한 관련 필드/차원을 나타냅니다. 사용자 지정 패널을 구성할 때 전체 테이블은 `wandb.log()`(`custom_data_table` 아래)에 전달된 명명된 키를 통해 액세스할 수 있으며 개별 필드는 열 이름 (`x`, `y` 및 `z`)을 통해 액세스할 수 있습니다. experiment 전체에서 여러 시간 단계에서 테이블을 기록할 수 있습니다. 각 테이블의 최대 크기는 10,000행입니다.

[Google Colab에서 사용해 보기](https://tiny.cc/custom-charts)

```python
# 데이터의 사용자 지정 테이블 기록
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

## 차트 사용자 정의

새 사용자 지정 차트를 추가하여 시작한 다음 표시되는 run에서 데이터를 선택하도록 쿼리를 편집합니다. 쿼리는 [GraphQL](https://graphql.org)을 사용하여 run의 config, summary 및 history 필드에서 데이터를 가져옵니다.

{{< img src="/images/app_ui/customize_chart.gif" alt="새 사용자 지정 차트를 추가한 다음 쿼리 편집" max=width="90%" >}}

### 사용자 지정 시각화

오른쪽 상단 모서리에서 **차트**를 선택하여 기본 사전 설정으로 시작합니다. 다음으로 **차트 필드**를 선택하여 쿼리에서 가져오는 데이터를 차트의 해당 필드에 매핑합니다. 다음은 쿼리에서 가져올 메트릭을 선택한 다음 아래 막대 차트 필드에 매핑하는 예입니다.

{{< img src="/images/app_ui/demo_make_a_custom_chart_bar_chart.gif" alt="프로젝트의 run에서 정확도를 보여주는 사용자 지정 막대 차트 만들기" max-width="90%" >}}

### Vega를 편집하는 방법

패널 상단의 **편집**을 클릭하여 [Vega](https://vega.github.io/vega/) 편집 모드로 들어갑니다. 여기에서 UI에서 대화형 차트를 만드는 [Vega 사양](https://vega.github.io/vega/docs/specification/)을 정의할 수 있습니다. 차트의 모든 측면을 변경할 수 있습니다. 예를 들어 제목을 변경하고 다른 색 구성표를 선택하고 곡선을 연결된 선이 아닌 일련의 점으로 표시할 수 있습니다. Vega 변환을 사용하여 값 배열을 히스토그램으로 bin하는 것과 같이 데이터 자체를 변경할 수도 있습니다. 패널 미리 보기가 대화식으로 업데이트되므로 Vega 사양 또는 쿼리를 편집할 때 변경 사항의 효과를 확인할 수 있습니다. [Vega 설명서 및 튜토리얼](https://vega.github.io/vega/)을 참조하세요.

**필드 참조**

W&B에서 차트로 데이터를 가져오려면 Vega 사양의 아무 곳에나 `"${field:<field-name>}"` 형식의 템플릿 문자열을 추가합니다. 그러면 오른쪽의 **차트 필드** 영역에 드롭다운이 생성되어 사용자가 쿼리 결과 열을 선택하여 Vega에 매핑할 수 있습니다.

필드의 기본값을 설정하려면 다음 구문을 사용합니다. `"${field:<field-name>:<placeholder text>}"`

### 차트 사전 설정 저장

모달 하단의 버튼을 사용하여 특정 시각화 패널에 대한 변경 사항을 적용합니다. 또는 Vega 사양을 저장하여 프로젝트의 다른 곳에서 사용할 수 있습니다. 재사용 가능한 차트 정의를 저장하려면 Vega 편집기 상단에서 **다른 이름으로 저장**을 클릭하고 사전 설정 이름을 지정합니다.

## 아티클 및 가이드

1. [W&B 기계 학습 시각화 IDE](https://wandb.ai/wandb/posts/reports/The-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)
2. [사용자 지정 차트를 사용하여 NLP Attention 기반 모델 시각화](https://wandb.ai/kylegoyette/gradientsandtranslation2/reports/Visualizing-NLP-Attention-Based-Models-Using-Custom-Charts--VmlldzoyNjg2MjM)
3. [사용자 지정 차트를 사용하여 Attention이 그래디언트 흐름에 미치는 영향 시각화](https://wandb.ai/kylegoyette/gradientsandtranslation/reports/Visualizing-The-Effect-of-Attention-on-Gradient-Flow-Using-Custom-Charts--VmlldzoyNjg1NDg)
4. [임의 곡선 기록](https://wandb.ai/stacey/presets/reports/Logging-Arbitrary-Curves--VmlldzoyNzQyMzA)

## 자주 묻는 질문

### 곧 제공 예정

* **폴링**: 차트의 데이터 자동 새로 고침
* **샘플링**: 효율성을 위해 패널에 로드된 총 포인트 수를 동적으로 조정

### 주의 사항

* 차트를 편집할 때 쿼리에서 예상한 데이터가 보이지 않습니까? 찾고 있는 열이 선택한 run에 기록되지 않았기 때문일 수 있습니다. 차트를 저장하고 run 테이블로 돌아가서 **눈** 아이콘으로 시각화하려는 run을 선택합니다.

## 일반적인 유스 케이스

* 오류 막대가 있는 막대 플롯 사용자 정의
* 사용자 지정 x-y 좌표가 필요한 모델 유효성 검사 메트릭 표시 (예: PR 곡선)
* 두 개의 다른 모델/experiment의 데이터 분포를 히스토그램으로 오버레이
* 트레이닝 중 여러 지점에서 스냅샷을 통해 메트릭의 변경 사항 표시
* W&B에서 아직 사용할 수 없는 고유한 시각화 만들기 (그리고 전 세계와 공유하기를 바랍니다)
