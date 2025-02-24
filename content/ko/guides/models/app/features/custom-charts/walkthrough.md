---
title: 'Tutorial: Use custom charts'
description: W&B UI에서 사용자 정의 차트 기능을 사용하는 방법에 대한 튜토리얼
menu:
  default:
    identifier: ko-guides-models-app-features-custom-charts-walkthrough
    parent: custom-charts
---

커스텀 차트를 사용하여 패널에 로드하는 데이터와 해당 시각화를 제어하세요.

## 1. W&B에 데이터 로깅하기

먼저 스크립트에서 데이터를 로깅합니다. 트레이닝 시작 시 설정된 단일 포인트(예: 하이퍼파라미터)에는 [wandb.config]({{< relref path="/guides/models/track/config.md" lang="ko" >}})를 사용하세요. 시간이 지남에 따라 여러 포인트를 위해서는 [wandb.log()]({{< relref path="/guides/models/track/log/" lang="ko" >}})를 사용하고, `wandb.Table()`을 사용하여 커스텀 2D 배열을 로깅하세요. 로그된 키당 최대 10,000개의 데이터 포인트를 로깅하는 것이 좋습니다.

```python
# 커스텀 데이터 테이블 로깅
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
  {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

[빠른 예제 노트북](https://bit.ly/custom-charts-colab)을 사용해 데이터 테이블을 로깅하고, 다음 단계에서는 커스텀 차트를 설정합니다. [라이브 리포트](https://app.wandb.ai/demo-team/custom-charts/reports/Custom-Charts--VmlldzoyMTk5MDc)에서 결과 차트가 어떻게 보이는지 확인하세요.

## 2. 쿼리 생성하기

시각화할 데이터를 로깅했으면 프로젝트 페이지로 이동하여 **`+`** 버튼을 클릭하여 새 패널을 추가한 다음 **Custom Chart**를 선택합니다. [이 워크스페이스](https://app.wandb.ai/demo-team/custom-charts)에서 따라 할 수 있습니다.

{{< img src="/images/app_ui/create_a_query.png" alt="구성할 준비가 된 새로운 빈 커스텀 차트" >}}

### 쿼리 추가하기

1. `summary`를 클릭하고 `historyTable`을 선택하여 run 기록에서 데이터를 가져오는 새 쿼리를 설정합니다.
2. `wandb.Table()`을 로깅한 키를 입력합니다. 위의 코드 조각에서는 `my_custom_table`입니다. [예제 노트북](https://bit.ly/custom-charts-colab)에서 키는 `pr_curve` 및 `roc_curve`입니다.

### Vega 필드 설정하기

이제 쿼리가 이러한 열을 로드하므로 Vega 필드 드롭다운 메뉴에서 선택할 수 있는 옵션으로 사용할 수 있습니다.

{{< img src="/images/app_ui/set_vega_fields.png" alt="쿼리 결과에서 열을 가져와 Vega 필드 설정" >}}

* **x-axis:** runSets_historyTable_r (recall)
* **y-axis:** runSets_historyTable_p (precision)
* **color:** runSets_historyTable_c (class label)

## 3. 차트 커스터마이징하기

이제 꽤 좋아 보이지만 산점도에서 선 그림으로 전환하고 싶습니다. **Edit**를 클릭하여 이 내장 차트에 대한 Vega 사양을 변경합니다. [이 워크스페이스](https://app.wandb.ai/demo-team/custom-charts)에서 따라 하세요.

{{< img src="/images/general/custom-charts-1.png" alt="" >}}

시각화를 커스터마이징하기 위해 Vega 사양을 업데이트했습니다.

* 플롯, 범례, x축 및 y축에 대한 제목 추가 (각 필드에 대해 "title" 설정)
* "mark" 값을 "point"에서 "line"으로 변경
* 사용하지 않는 "size" 필드 제거

{{< img src="/images/app_ui/customize_vega_spec_for_pr_curve.png" alt="" >}}

이것을 이 프로젝트의 다른 곳에서 사용할 수 있는 사전 설정으로 저장하려면 페이지 상단의 **Save as**를 클릭합니다. 다음은 ROC 곡선과 함께 결과가 어떻게 보이는지입니다.

{{< img src="/images/general/custom-charts-2.png" alt="" >}}

## 보너스: 합성 히스토그램

히스토그램은 숫자 분포를 시각화하여 더 큰 데이터셋을 이해하는 데 도움이 될 수 있습니다. 합성 히스토그램은 동일한 bin에서 여러 분포를 보여 주어 서로 다른 모델 또는 모델 내의 서로 다른 클래스에서 두 개 이상의 메트릭을 비교할 수 있습니다. 운전 장면에서 오브젝트를 감지하는 시멘틱 세그멘테이션 모델의 경우 정확도 대 IOU (intersection over union)에 대한 최적화의 효과를 비교하거나 서로 다른 모델이 자동차 (데이터에서 크고 일반적인 영역) 대 교통 표지판 (훨씬 작고 덜 일반적인 영역)을 얼마나 잘 감지하는지 알고 싶을 수 있습니다. [데모 Colab](https://bit.ly/custom-charts-colab)에서는 10가지 생물 클래스 중 두 가지에 대한 신뢰도 점수를 비교할 수 있습니다.

{{< img src="/images/app_ui/composite_histograms.png" alt="" >}}

커스텀 합성 히스토그램 패널의 자체 버전을 만들려면:

1. 워크스페이스 또는 Report에서 새 Custom Chart 패널을 만듭니다 ("Custom Chart" 시각화를 추가하여). 오른쪽 상단의 "Edit" 버튼을 눌러 내장 패널 유형에서 시작하여 Vega 사양을 수정합니다.
2. 해당 내장 Vega 사양을 [Vega의 합성 히스토그램에 대한 MVP 코드](https://gist.github.com/staceysv/9bed36a2c0c2a427365991403611ce21)로 바꿉니다. 이 Vega 사양에서 직접 메인 제목, 축 제목, 입력 도메인 및 기타 세부 정보를 수정할 수 있습니다 [Vega 구문 사용](https://vega.github.io/) (색상을 변경하거나 세 번째 히스토그램을 추가할 수도 있습니다 :)
3. 오른쪽 쿼리를 수정하여 wandb 로그에서 올바른 데이터를 로드합니다. `summaryTable` 필드를 추가하고 해당 `tableKey`를 `class_scores`로 설정하여 run에서 로깅한 `wandb.Table`을 가져옵니다. 이렇게 하면 드롭다운 메뉴를 통해 `class_scores`로 로깅된 `wandb.Table`의 열이 있는 두 개의 히스토그램 bin 세트 (`red_bins` 및 `blue_bins`)를 채울 수 있습니다. 내 예제에서는 빨간색 bin에 대한 `animal` 클래스 예측 점수와 파란색 bin에 대한 `plant`를 선택했습니다.
4. 미리보기 렌더링에서 보이는 플롯에 만족할 때까지 Vega 사양 및 쿼리를 계속 변경할 수 있습니다. 완료되면 상단의 **Save as**를 클릭하고 커스텀 플롯에 이름을 지정하여 재사용할 수 있습니다. 그런 다음 **Apply from panel library**를 클릭하여 플롯을 완료합니다.

다음은 매우 간단한 실험에서 얻은 결과입니다. 하나의 에포크에 대해 1000개의 예제만 트레이닝하면 대부분의 이미지가 식물이 아니라고 매우 확신하고 어떤 이미지가 동물일 수 있는지 매우 불확실한 모델이 생성됩니다.

{{< img src="/images/general/custom-charts-3.png" alt="" >}}

{{< img src="/images/general/custom-charts-4.png" alt="" >}}
