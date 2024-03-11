---
description: Tutorial of using the custom charts feature in the W&B UI
displayed_sidebar: default
---

# 맞춤형 차트 안내서

W&B의 기본 차트를 넘어서 맞춤형 차트 기능을 사용하여 패널에 로딩되는 데이터의 세부 사항을 정확히 제어하고 데이터를 시각화하는 방법을 설정하세요.

**개요**

1. W&B에 데이터 로그하기
2. 쿼리 생성하기
3. 차트 사용자 정의하기

## 1. W&B에 데이터 로그하기

먼저 스크립트에서 데이터를 로그하세요. 트레이닝 시작 시 설정되는 단일 포인트, 예를 들어 하이퍼파라미터에는 [wandb.config](../../../../guides/track/config.md)를 사용하세요. 시간이 지남에 따라 여러 포인트에 대해서는 [wandb.log()](../../../../guides/track/log/intro.md)를 사용하고, wandb.Table()를 사용하여 맞춤 2D 배열을 로그하세요. 로그된 각 키당 최대 10,000개의 데이터 포인트를 로그하는 것이 좋습니다.

```python
# 데이터의 맞춤 테이블 로깅
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

[빠른 예제 노트북](https://bit.ly/custom-charts-colab)을 시도하여 데이터 테이블을 로그하고, 다음 단계에서는 맞춤형 차트를 설정합니다. [라이브 리포트](https://app.wandb.ai/demo-team/custom-charts/reports/Custom-Charts--VmlldzoyMTk5MDc)에서 결과 차트를 확인하세요.

## 2. 쿼리 생성하기

시각화할 데이터를 로그한 후 프로젝트 페이지로 이동하여 **`+`** 버튼을 클릭하여 새 패널을 추가한 다음 **맞춤형 차트**를 선택하세요. [이 워크스페이스](https://app.wandb.ai/demo-team/custom-charts)에서 따라할 수 있습니다.

![구성할 준비가 된 새로운 빈 맞춤형 차트](/images/app_ui/create_a_query.png)

### 쿼리 추가하기

1. `summary`를 클릭하고 `historyTable`을 선택하여 실행 이력에서 데이터를 가져오는 새 쿼리를 설정합니다.
2. **wandb.Table()**을 로그한 키를 입력하세요. 위의 코드 조각에서는 `my_custom_table`이었습니다. [예제 노트북](https://bit.ly/custom-charts-colab)에서는 키가 `pr_curve`와 `roc_curve`입니다.

### Vega 필드 설정하기

이제 이 열이 쿼리에 로드되어 Vega 필드 드롭다운 메뉴에서 선택할 수 있는 옵션으로 제공됩니다:

![쿼리 결과에서 열을 가져와 Vega 필드를 설정하기](/images/app_ui/set_vega_fields.png)

* **x축:** runSets\_historyTable\_r (재현율)
* **y축:** runSets\_historyTable\_p (정밀도)
* **색상:** runSets\_historyTable\_c (클래스 라벨)

## 3. 차트 사용자 정의하기

이제 꽤 괜찮아 보이지만, 산점도에서 선형 차트로 전환하고 싶습니다. 내장 차트의 Vega 사양을 변경하려면 **편집**을 클릭하세요. [이 워크스페이스](https://app.wandb.ai/demo-team/custom-charts)에서 따라하세요.

시각화를 사용자 정의하기 위해 Vega 사양을 업데이트했습니다:

* 플롯, 범례, x축 및 y축에 대한 제목 추가 (각 필드에 “title” 설정)
* “mark”의 값을 “point”에서 “line”으로 변경
* 사용되지 않는 “size” 필드 제거

이 프로젝트의 다른 곳에서 사용할 수 있도록 사전 설정으로 저장하려면 페이지 상단의 **Save as**를 클릭하세요. 다음은 결과와 ROC 곡선입니다:

## 보너스: 복합 히스토그램

히스토그램은 수치 분포를 시각화하여 더 큰 데이터셋을 이해하는 데 도움을 줍니다. 복합 히스토그램은 같은 구간에 걸쳐 여러 분포를 보여주어, 다른 모델 또는 모델 내 다른 클래스 간에 두 개 이상의 메트릭을 비교할 수 있게 합니다. 운전 장면에서 오브젝트를 감지하는 시멘틱 세그멘테이션 모델의 경우, 정확도 대비 IOU (intersection over union)를 최적화하는 효과를 비교하거나, 다른 모델이 차량(데이터에서 크고 흔한 영역) 대 교통 표지판(훨씬 더 작고 덜 흔한 영역)을 얼마나 잘 감지하는지 알고 싶을 수 있습니다. [데모 Colab](https://bit.ly/custom-charts-colab)에서는 생물의 열 가지 클래스 중 두 클래스의 신뢰 점수를 비교할 수 있습니다.

맞춤형 복합 히스토그램 패널의 버전을 만드는 방법: