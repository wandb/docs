---
title: 커스텀 차트
cascade:
- url: guides/app/features/custom-charts/:filename
menu:
  default:
    identifier: ko-guides-models-app-features-custom-charts-_index
    parent: w-b-app-ui-reference
url: guides/app/features/custom-charts
weight: 2
---

W&B 프로젝트에서 커스텀 차트를 만들어보세요. 임의의 테이블 데이터를 로그하고 원하는 방식대로 시각화할 수 있습니다. [Vega](https://vega.github.io/vega/)의 강력한 기능으로 폰트, 색상, 툴팁 등 세부 사항을 직접 컨트롤할 수 있습니다.

* 코드: 예제 [Colab 노트북](https://tiny.cc/custom-charts)을 실행해보세요.
* 영상: [워크스루 영상](https://www.youtube.com/watch?v=3-N9OV6bkSM) 시청하기.
* 예제: 빠르게 Keras와 Sklearn을 확인할 수 있는 [데모 노트북](https://colab.research.google.com/drive/1g-gNGokPWM2Qbc8p1Gofud0_5AoZdoSD?usp=sharing)

{{< img src="/images/app_ui/supported_charts.png" alt="vega.github.io/vega에서 지원하는 차트" max-width="90%" >}}

### 작동 방식

1. **데이터 로그**: 스크립트에서 [config]({{< relref path="/guides/models/track/config.md" lang="ko" >}}) 및 summary 데이터를 로그하세요.
2. **차트 커스터마이즈**: [GraphQL](https://graphql.org) 쿼리로 로그된 데이터를 불러옵니다. 쿼리 결과를 강력한 시각화 언어인 [Vega](https://vega.github.io/vega/)로 시각화할 수 있습니다.
3. **차트 로그**: `wandb.plot_table()`을 이용해 직접 만든 프리셋을 스크립트에서 호출하세요.

{{< img src="/images/app_ui/pr_roc.png" alt="PR 및 ROC 곡선" >}}

원하는 데이터가 보이지 않는다면, 찾는 컬럼이 선택한 Runs에 로그되지 않았을 수 있습니다. 차트를 저장하고 Runs 테이블로 돌아가 **eye** 아이콘으로 선택된 Runs를 확인하세요.


## 스크립트에서 차트 로그하기

### 내장 프리셋

W&B에는 스크립트에서 바로 로그할 수 있는 여러 기본 차트 프리셋이 있습니다. 라인 플롯, 산점도, 바 차트, 히스토그램, PR 곡선, ROC 곡선이 포함되어 있습니다.

{{< tabpane text=true >}}
{{% tab header="Line plot" value="line-plot" %}}

  `wandb.plot.line()`

  커스텀 라인 플롯을 로그하세요—(x,y)로 연결된 순서쌍 포인트 리스트를 임의 축 x와 y에 표시합니다.

  ```python
  with wandb.init() as run:
    data = [[x, y] for (x, y) in zip(x_values, y_values)]
    table = wandb.Table(data=data, columns=["x", "y"])
    run.log(
        {
            "my_custom_plot_id": wandb.plot.line(
                table, "x", "y", title="Custom Y vs X Line Plot"
            )
        }
    )
  ```

  라인 플롯은 임의의 2차원에서 곡선을 로그합니다. 두 리스트의 값을 서로 플롯할 때, 리스트의 값 개수가 정확히 일치해야 합니다(예: 각 포인트는 x와 y를 모두 가져야 함).

  {{< img src="/images/app_ui/line_plot.png" alt="커스텀 라인 플롯" >}}

  [예제 report 보기](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA) 또는 [Google Colab 노트북 예제 실행](https://tiny.cc/custom-charts).

{{% /tab %}}

{{% tab header="Scatter plot" value="scatter-plot" %}}

  `wandb.plot.scatter()`

  커스텀 산점도(Scatter plot)를 로그하세요—임의 축 쌍 x와 y에 (x, y) 포인트 리스트를 표시합니다.

  ```python
  with wandb.init() as run:
    data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
    table = wandb.Table(data=data, columns=["class_x", "class_y"])
    run.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
  ```

  임의의 두 차원에서 산점도를 로그하는 데 사용할 수 있습니다. 두 리스트의 값을 서로 플롯하려면, 리스트 값의 개수가 정확히 일치해야 합니다(예: 각각이 x, y를 가져야 함).

  {{< img src="/images/app_ui/demo_scatter_plot.png" alt="산점도" >}}

  [예제 report 보기](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ) 또는 [Google Colab 노트북 예제 실행](https://tiny.cc/custom-charts).

{{% /tab %}}

{{% tab header="Bar chart" value="bar-chart" %}}

  `wandb.plot.bar()`

  커스텀 바 차트를 로그하세요—라벨이 붙은 값 리스트를 바 형태로 간단하게 로그할 수 있습니다:

  ```python
  with wandb.init() as run:
    data = [[label, val] for (label, val) in zip(labels, values)]
    table = wandb.Table(data=data, columns=["label", "value"])
    run.log(
        {
            "my_bar_chart_id": wandb.plot.bar(
                table, "label", "value", title="Custom Bar Chart"
            )
        }
    )
  ```

  임의의 바 차트를 로그하는 데 사용할 수 있습니다. 라벨과 값 리스트의 개수가 정확히 일치해야 합니다(예: 모든 데이터포인트에 둘 다 필요).

{{< img src="/images/app_ui/demo_bar_plot.png" alt="데모 바 플롯" >}}

  [예제 report 보기](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk) 또는 [Google Colab 노트북 예제 실행](https://tiny.cc/custom-charts).
{{% /tab %}}

{{% tab header="Histogram" value="histogram" %}}

  `wandb.plot.histogram()`

  커스텀 히스토그램을 몇 줄의 코드로 로그하세요—값 리스트를 카운트/빈도로 bin에 정렬합니다. 예를 들어 예측 신뢰도 점수(`scores`) 리스트의 분포를 시각화한다고 가정해봅시다:

  ```python
  with wandb.init() as run:
    data = [[s] for s in scores]
    table = wandb.Table(data=data, columns=["scores"])
    run.log({"my_histogram": wandb.plot.histogram(table, "scores", title=None)})
  ```

  임의의 히스토그램을 로그할 수 있으며, `data`는 2차원 행렬(행/열)을 지원하므로 리스트의 리스트 형태입니다.

  {{< img src="/images/app_ui/demo_custom_chart_histogram.png" alt="커스텀 히스토그램" >}}

  [예제 report 보기](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM) 또는 [Google Colab 노트북 예제 실행](https://tiny.cc/custom-charts).

{{% /tab %}}

{{% tab header="PR curve" value="pr-curve" %}}

  `wandb.plot.pr_curve()`

  [Precision-Recall 곡선](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)을 한 줄로 생성하세요:

  ```python
  with wandb.init() as run:
    plot = wandb.plot.pr_curve(ground_truth, predictions, labels=None, classes_to_plot=None)

    run.log({"pr": plot})
  ```

  아래 요건을 충족할 때 언제든 사용 가능합니다:

  * 모델이 예제 집합에 대해 예측한 점수(`predictions`)
  * 해당 예제의 그라운드 트루스 라벨(`ground_truth`)
  * (선택) 라벨/클래스명 리스트 (`labels=["cat", "dog", "bird"...]` 같이 label 인덱스 0=cat, 1=dog, 2=새, ...)
  * (선택) 플롯에 시각화할 라벨의 서브셋(리스트 형태)

  {{< img src="/images/app_ui/demo_average_precision_lines.png" alt="Precision-recall 곡선" >}}


  [예제 report 보기](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY) 또는 [Google Colab 노트북 예제 실행](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing).

{{% /tab %}}

{{% tab header="ROC curve" value="roc-curve" %}}

  `wandb.plot.roc_curve()`

  [ROC 곡선](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)을 한 줄로 생성하세요:

  ```python
  with wandb.init() as run:
    # ground_truth는 실제 라벨 리스트, predictions는 예측 점수 리스트입니다
    ground_truth = [0, 1, 0, 1, 0, 1]
    predictions = [0.1, 0.4, 0.35, 0.8, 0.7, 0.9]

    # ROC 곡선 플롯 생성
    # labels는 옵션 클래스명, classes_to_plot은 볼 라벨의 서브셋(옵션)
    plot = wandb.plot.roc_curve(
        ground_truth, predictions, labels=None, classes_to_plot=None
    )

    run.log({"roc": plot})
  ```

  다음 조건이 맞으면 언제든 로그할 수 있습니다:

  * 모델 예측 점수 리스트 (`predictions`)
  * 해당 예제의 그라운드 트루스 라벨(`ground_truth`)
  * (선택) 클래스명 리스트 (`labels=["cat", "dog", "bird"...]`)
  * (선택) 플롯에 시각화할 라벨의 서브셋(리스트 형태)

  {{< img src="/images/app_ui/demo_custom_chart_roc_curve.png" alt="ROC 곡선" >}}

  [예제 report 보기](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE) 또는 [Google Colab 노트북 예제 실행](https://colab.research.google.com/drive/1_RMppCqsA8XInV_jhJz32NCZG6Z5t1RO?usp=sharing).

{{% /tab %}}
{{< /tabpane >}}

### 커스텀 프리셋

기본 프리셋을 원하는 대로 수정하거나 완전히 새로운 프리셋을 저장해 차트로 활용할 수 있습니다. 차트 ID를 이용해 해당 커스텀 프리셋으로 데이터를 직접 스크립트에서 로그하세요. [Google Colab 노트북 예제 실행](https://tiny.cc/custom-charts).

```python
# 플롯할 테이블을 생성
table = wandb.Table(data=data, columns=["step", "height"])

# 테이블 컬럼과 차트 필드 매핑
fields = {"x": "step", "value": "height"}

# 테이블로 새로운 커스텀 차트 프리셋 활용
# 직접 저장한 차트 프리셋 사용시 vega_spec_name을 변경하세요
my_custom_chart = wandb.plot_table(
    vega_spec_name="carey/new_chart",
    data_table=table,
    fields=fields,
)
```

{{< img src="/images/app_ui/custom_presets.png" alt="커스텀 차트 프리셋" max-width="90%" >}}

## 데이터 로그하기

스크립트에서 아래 데이터 타입들을 로그할 수 있으며, 이를 커스텀 차트에서 활용할 수 있습니다:

* **Config**: 실험의 초기 설정값(독립 변수 역할). 트레이닝 시작 시 `wandb.Run.config`에 키-값 형태로 저장한 필드들이 해당됩니다. 예: `wandb.Run.config.learning_rate = 0.0001`
* **Summary**: 트레이닝 중에 로그된 단일 값(결과/종속 변수로 사용). 예: `wandb.Run.log({"val_acc": 0.8})`. 트레이닝 중 같은 키로 여러 번 쓰면 마지막 값이 summary에 저장됩니다.
* **History**: 로그된 스칼라의 전체 시계열 데이터는 `history` 필드로 쿼리할 수 있습니다.
* **summaryTable**: 여러 값의 리스트를 로그하려면 `wandb.Table()`로 데이터를 저장한 뒤, 커스텀 패널에서 쿼리하세요.
* **historyTable**: 히스토리 데이터를 보려면, 커스텀 차트 패널에서 `historyTable`을 쿼리하세요. `wandb.Table()`을 호출하거나 커스텀 차트를 로그할 때마다 해당 step에 새로운 테이블이 생성됩니다.

### 커스텀 테이블 로그 방법

`wandb.Table()`을 활용하면 데이터를 2D 배열 형식으로 로그할 수 있습니다. 보통 테이블의 한 행은 하나의 데이터포인트를 의미하며, 각 열은 시각화하고 싶은 해당 데이터의 필드/차원을 나타냅니다. 커스텀 패널 구성 시, 이 테이블은 `wandb.Run.log()`에 전달한 key 이름(`custom_data_table` 예시)으로 접근 가능하고, 각 필드는 컬럼명(`x`, `y`, `z`)으로 접근할 수 있습니다. 실험 도중 여러 시간 step에 걸쳐 테이블을 로그할 수 있습니다. 테이블 최대 크기는 10,000행입니다. [Google Colab 예제 실행](https://tiny.cc/custom-charts).

```python
with wandb.init() as run:
  # 데이터를 커스텀 테이블로 로그
  my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
  run.log(
      {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
  )
```

## 차트 커스터마이즈하기

새로운 커스텀 차트를 추가해 시작해보세요. 쿼리를 수정해 원하는 Runs에서 데이터를 선택할 수 있습니다. 쿼리는 [GraphQL](https://graphql.org)을 사용하며 config, summary, history 필드에서 데이터를 가져옵니다.

{{< img src="/images/app_ui/customize_chart.gif" alt="커스텀 차트 생성" max=width="90%" >}}

### 커스텀 시각화

오른쪽 상단에서 **차트(Chart)**를 선택해 기본 프리셋으로 시작할 수 있습니다. 그다음 **차트 필드(Chart fields)**를 골라 쿼리에서 불러온 데이터를 차트의 필드에 맵핑하세요.

아래 이미지는 메트릭을 선택해 바 차트 필드에 맵핑하는 예시를 보여줍니다.

{{< img src="/images/app_ui/demo_make_a_custom_chart_bar_chart.gif" alt="커스텀 바 차트 생성" max-width="90%" >}}

### Vega 편집 방법

패널 상단의 **Edit** 버튼을 클릭해 [Vega](https://vega.github.io/vega/) 편집 모드로 진입하세요. 여기서 [Vega 스펙](https://vega.github.io/vega/docs/specification/)을 정의해 대화형 차트를 만들 수 있습니다. 차트의 제목, 색상 스킴, 또는 곡선을 점 형태로 표시하는 등 원하는 모든 요소를 바꿀 수 있습니다. Vega 변환(transform) 기능을 활용해 배열 값을 히스토그램으로 binning하는 등 데이터 자체도 가공할 수 있습니다. 패널의 미리보기는 실시간으로 업데이트되어 Vega 스펙/쿼리 변경 시 즉시 효과를 확인할 수 있습니다. 자세한 내용은 [Vega 공식 문서와 튜토리얼](https://vega.github.io/vega/)을 참고하세요.

**필드 참조**

W&B에서 차트로 데이터를 불러오려면 Vega 스펙 내 어디서든 `"${field:<필드명>}"` 형태의 템플릿 문자열을 추가하세요. 그러면 **Chart fields** 영역에 해당 컬럼을 Vega에 맵핑할 수 있는 드롭다운이 생깁니다.

필드의 기본값을 지정하려면: `"${field:<필드명>:<placeholder 텍스트>}"`

### 차트 프리셋 저장

모달 하단 버튼을 통해 특정 시각화 패널 변경사항을 적용할 수 있습니다. 또는 Vega 스펙을 프로젝트 다른 곳에서 재사용할 수 있도록 저장할 수도 있습니다. 재사용 가능한 차트 정의를 저장하려면 Vega 편집기 상단 **Save as**를 클릭하고 프리셋 이름을 지정하세요.

## 관련 아티클 및 가이드

1. [The W&B Machine Learning Visualization IDE](https://wandb.ai/wandb/posts/reports/The-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg)
2. [Visualizing NLP Attention Based Models](https://wandb.ai/kylegoyette/gradientsandtranslation2/reports/Visualizing-NLP-Attention-Based-Models-Using-Custom-Charts--VmlldzoyNjg2MjM)
3. [Visualizing The Effect of Attention on Gradient Flow](https://wandb.ai/kylegoyette/gradientsandtranslation/reports/Visualizing-The-Effect-of-Attention-on-Gradient-Flow-Using-Custom-Charts--VmlldzoyNjg1NDg)
4. [Logging arbitrary curves](https://wandb.ai/stacey/presets/reports/Logging-Arbitrary-Curves--VmlldzoyNzQyMzA)


## 주요 유스 케이스

* 에러 바가 포함된 바 플롯 커스터마이즈
* (precision-recall curves와 같이) 커스텀 x-y 좌표가 필요한 모델 검증 메트릭 시각화
* 서로 다른 두 모델/Experiments의 데이터 분포를 히스토그램으로 비교
* 트레이닝 중 여러 시점의 스냅샷을 통해 시계열로 메트릭 변화 보여주기
* 아직 W&B에서 지원하지 않는 독특한 시각화 생성(다른 사람과 공유도 가능!)