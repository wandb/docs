---
title: Tutorial: Use custom charts
description: W&B UI에서 커스텀 차트 기능 사용 튜토리얼
displayed_sidebar: default
---

사용자 정의 차트를 사용하여 패널에 로딩되는 데이터를 제어하고 그 시각화를 관리하세요.

**개요**

1. 데이터 W&B에 로그하기
2. 쿼리 생성하기
3. 차트 사용자 정의하기

## 1. 데이터 W&B에 로그하기

먼저, 스크립트에서 데이터를 로그하세요. 트레이닝 시작 시 설정되는 단일 포인트에는 [wandb.config](../../../../guides/track/config.md)를 사용하고, 시간이 흐르며 여러 포인트를 로그하거나 사용자 정의 2D 배열을 wandb.Table()로 로그할 때는 [wandb.log()](../../../../guides/track/log/intro.md)를 사용하세요. 로그당 최대 10,000개의 데이터 포인트를 로그하는 것을 권장합니다.

```python
# 데이터의 사용자 정의 테이블 로그하기
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

데이터 테이블을 로그하기 위한 [빠른 예제 노트북 시도해보기](https://bit.ly/custom-charts-colab), 그리고 다음 단계에서 사용자 정의 차트를 설정해보겠습니다. 생성된 차트가 어떻게 보이는지 [실시간 리포트](https://app.wandb.ai/demo-team/custom-charts/reports/Custom-Charts--VmlldzoyMTk5MDc)에서 확인하세요.

## 2. 쿼리 생성하기

데이터를 시각화하기 위해 로그한 후, 프로젝트 페이지로 이동하여 새로운 패널을 추가하는 **`+`** 버튼을 클릭하고 **Custom Chart**를 선택하세요. [이 워크스페이스](https://app.wandb.ai/demo-team/custom-charts)에서 따를 수 있습니다.

![새로운, 비어 있는 사용자 정의 차트가 구성 준비 완료](/images/app_ui/create_a_query.png)

### 쿼리 추가하기

1. `summary`를 클릭하고 `historyTable`을 선택하여 run 기록에서 데이터를 가져오는 새로운 쿼리를 설정합니다.
2. **wandb.Table()**을 로그한 키를 입력하세요. 위의 코드조각에서는 `my_custom_table`이었습니다. [예제 노트북](https://bit.ly/custom-charts-colab)에서는 키가 `pr_curve`와 `roc_curve`입니다.

### Vega 필드 설정하기

이제 쿼리가 이러한 열을 로딩하고 있으므로, Vega 필드 드롭다운 메뉴에서 옵션으로 선택할 수 있게 됩니다:

![쿼리 결과에서 Vega 필드를 설정하기 위해 열 가져오기](/images/app_ui/set_vega_fields.png)

* **x축:** runSets_historyTable_r (recall)
* **y축:** runSets_historyTable_p (precision)
* **색상:** runSets_historyTable_c (클래스 라벨)

## 3. 차트 사용자 정의하기

눈에 그럴듯하게 보이지만, 산점도에서 선형 차트로 변경하고 싶습니다. 이 기본 차트에 대한 Vega 사양을 변경하려면 **Edit**을 클릭하세요. [이 워크스페이스](https://app.wandb.ai/demo-team/custom-charts)에서 따라가기.

![](/images/general/custom-charts-1.png)

시각화를 사용자 정의하기 위해 Vega 사양을 업데이트했습니다:

* 플롯, 범례, x축, y축에 제목 추가(각 필드에 “title” 설정)
* “mark”의 값을 “point”에서 “line”으로 변경
* 사용되지 않는 “size” 필드 제거

![](/images/app_ui/customize_vega_spec_for_pr_curve.png)

이 프로젝트에서 다른 곳에서도 사용할 수 있는 프리셋으로 저장하려면 페이지 상단의 **Save as**를 클릭하세요. 결과가 어떻게 보이는지와 ROC 곡선도 확인하세요:

![](/images/general/custom-charts-2.png)

## 보너스: 복합 히스토그램

히스토그램은 큰 데이터셋을 이해하는 데 도움을 주기 위해 수치 분포를 시각화할 수 있습니다. 복합 히스토그램은 같은 빈(bin)에 여러 분포를 보여주고, 두 개 이상의 메트릭을 서로 다른 모델 또는 모델 내에서 다양한 클래스에 걸쳐 비교할 수 있게 해줍니다. 운전 장면에서 오브젝트를 감지하는 시멘틱 세그멘테이션 모델의 경우, 정확도를 최적화하는 것과 intersection over union (IOU)를 최적화하는 것의 효율성을 비교할 수 있습니다. 혹은 다른 모델이 자동차(데이터에서 큰 일반적인 영역)을 탐지하는 것과 교통 표지판(훨씬 작은, 덜 일반적인 영역)을 탐지하는 성능을 알고 싶을 때도 있습니다. [데모 Colab](https://bit.ly/custom-charts-colab)에서, 생물 클래스의 두 가지를 비교할 수 있습니다.

![](/images/app_ui/composite_histograms.png)

사용자 정의 복합 히스토그램 패널을 만들기 위해:

1. 워크스페이스 또는 리포트에서 새 Custom Chart 패널을 만들고(“Custom Chart” 시각화를 추가), 모든 기본 패널 유형에서 시작되는 Vega 사양을 수정하려면 오른쪽 상단의 “Edit” 버튼을 클릭하세요.
2. 기본 Vega 사양을 내 [Vega에서의 복합 히스토그램에 대한 MVP 코드](https://gist.github.com/staceysv/9bed36a2c0c2a427365991403611ce21)로 교체하세요. Vega 구문을 사용하여 [Vega syntax](https://vega.github.io/)에서 메인 제목, 축 제목, 입력 도메인 및 기타 세부 정보를 직접 수정할 수 있습니다(색상을 변경하거나 세 번째 히스토그램을 추가할 수도 있습니다).
3. wandb 로그에서 올바른 데이터를 로드하기 위해 오른쪽의 쿼리를 수정하세요. “summaryTable” 필드를 추가하고 해당 “tableKey”를 “class_scores”로 설정하여 run에 의해 로그된 wandb.Table을 가져오세요. 이는 wandb.Table이 “class_scores”로 기록된 열을 통해 드롭다운 메뉴로 두 히스토그램 빈 세트(“red_bins”와 “blue_bins”)를 채울 수 있게 해줍니다. 제 예제에서는 “animal” 클래스 예측 점수를 빨간 빈으로, “plant”를 파란 빈으로 선택했습니다.
4. 검토 화면에서 볼 플롯에 만족할 때까지 Vega 사양과 쿼리를 계속 수정할 수 있습니다. 완료되면, 상단에서 “Save as”를 클릭하여 사용자 정의 플롯에 이름을 지정하고 재사용하세요. 그런 다음 “Apply from panel library”를 클릭하여 플롯 작성을 완료하세요.

여기 제 실험 결과가 있습니다: 1,000개의 예제만으로 1에포크에 대해 트레이닝된 모델은 대부분의 이미지가 식물은 아니라고 매우 자신있어 하며 어떤 이미지가 동물일지는 매우 불확실합니다.

![](/images/general/custom-charts-3.png)

![](/images/general/custom-charts-4.png)
