---
description: Tutorial of using the custom charts feature in the W&B UI
displayed_sidebar: default
---

# 커스텀 차트 가이드

W&B의 기본 차트를 넘어서고 싶다면, 새로운 **커스텀 차트** 기능을 사용하여 패널에 로드하는 데이터의 세부 사항을 정확히 제어하고 그 데이터를 어떻게 시각화할지 결정하세요.

**개요**

1. W&B에 데이터 로그하기
2. 쿼리 생성하기
3. 차트 커스터마이즈하기

## 1. W&B에 데이터 로그하기

먼저, 스크립트에서 데이터를 로그하세요. 학습 시작 시에 설정되는 하이퍼파라미터와 같이 단일 데이터 포인트의 경우 [wandb.config](../../../../guides/track/config.md)를 사용하세요. 시간이 지남에 따라 여러 데이터 포인트를 로그하려면 [wandb.log()](../../../../guides/track/log/intro.md)를 사용하고, wandb.Table()로 커스텀 2D 배열을 로그하세요. 로그된 키당 최대 10,000개의 데이터 포인트를 로그하는 것이 좋습니다.

```python
# 데이터 커스텀 테이블 로깅
my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
wandb.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
)
```

[데이터 테이블 로깅을 위한 빠른 예제 노트북](https://bit.ly/custom-charts-colab)을 시도해보고, 다음 단계에서 커스텀 차트를 설정하겠습니다. [라이브 리포트](https://app.wandb.ai/demo-team/custom-charts/reports/Custom-Charts--VmlldzoyMTk5MDc)에서 결과 차트를 확인해보세요.

## 2. 쿼리 생성하기

시각화할 데이터를 로그한 후, 프로젝트 페이지로 이동하여 **`+`** 버튼을 클릭하여 새 패널을 추가하고 **커스텀 차트**를 선택하세요. [이 워크스페이스](https://app.wandb.ai/demo-team/custom-charts)에서 따라할 수 있습니다.

![구성할 준비가 된 새로운 빈 커스텀 차트](/images/app_ui/create_a_query.png)

### 쿼리 추가하기

1. `summary`를 클릭하고 `historyTable`을 선택하여 실행 기록에서 데이터를 가져오는 새 쿼리를 설정합니다.
2. **wandb.Table()**을 로그한 키를 입력하세요. 위의 코드 조각에서는 `my_custom_table`이었습니다. [예제 노트북](https://bit.ly/custom-charts-colab)에서는 키가 `pr_curve`와 `roc_curve`입니다.

### Vega 필드 설정하기

이제 이러한 열이 쿼리에서 로드되면, Vega 필드 드롭다운 메뉴에서 선택할 수 있는 옵션으로 사용할 수 있습니다:

![쿼리 결과에서 열을 불러와 Vega 필드를 설정하기](/images/app_ui/set_vega_fields.png)

* **x축:** runSets\_historyTable\_r (재현율)
* **y축:** runSets\_historyTable\_p (정밀도)
* **색상:** runSets\_historyTable\_c (클래스 라벨)

## 3. 차트 커스터마이즈하기

이제 꽤 좋아 보이지만, 저는 산점도에서 선형 플롯으로 전환하고 싶습니다. 이 기본 차트에 대한 Vega spec을 변경하기 위해 **편집**을 클릭하세요. [이 워크스페이스](https://app.wandb.ai/demo-team/custom-charts)에서 계속 따라하세요.

시각화를 커스터마이즈하기 위해 Vega spec을 업데이트했습니다:

* 플롯, 범례, x축 및 y축에 대한 제목 추가 (각 필드에 대해 "title" 설정)
* “mark”의 값을 “point”에서 “line”으로 변경
* 사용하지 않는 “size” 필드 제거

이 프로젝트에서 다른 곳에 사용할 수 있는 프리셋으로 저장하려면 페이지 상단에서 **저장하기**를 클릭하세요. 여기에 ROC 곡선과 함께 결과가 어떻게 보이는지 나와 있습니다:

## 보너스: 복합 히스토그램

히스토그램은 수치적 분포를 시각화하여 우리가 더 큰 데이터세트를 이해하는 데 도움을 줄 수 있습니다. 복합 히스토그램은 같은 구간에 여러 분포를 보여주어, 다른 모델이나 모델 내 다른 클래스에 걸쳐 두 개 이상의 메트릭을 비교할 수 있게 합니다. 운전 장면에서 개체를 감지하는 semantic segmentation 모델의 경우, 정확도 대비 IOU(intersection over union) 최적화의 효과를 비교하거나, 다른 모델이 자동차(데이터에서 크고 일반적인 영역)와 교통 표지판(훨씬 더 작고 덜 일반적인 영역)을 얼마나 잘 감지하는지 알고 싶을 수 있습니다. [데모 Colab](https://bit.ly/custom-charts-colab)에서는 생물 10개 클래스 중 두 클래스의 신뢰도 점수를 비교할 수 있습니다.

커스텀 복합 히스토그램 패널의 자체 버전을 만들려면:

1. 워크스페이스 또는 리포트에 새로운 "커스텀 차트" 시각화를 추가하여 새 커스텀 차트 패널을 생성합니다. 오른쪽 상단의 "편집" 버튼을 눌러 기본 패널 유형에서 시작하여 Vega spec을 수정합니다.
2. 그 기본 Vega spec을 [Vega를 위한 복합 히스토그램의 MVP 코드](https://gist.github.com/staceysv/9bed36a2c0c2a427365991403611ce21)로 대체하세요. [Vega 문법](https://vega.github.io/)을 사용하여 이 Vega spec에서 메인 제목, 축 제목, 입력 도메인 및 기타 세부 정보를 직접 수정할 수 있습니다(색상을 변경하거나 세 번째 히스토그램을 추가할 수도 있습니다 :)
3. 오른쪽의 쿼리를 수정하여 wandb 로그에서 올바른 데이터를 로드하도록 합니다. "summaryTable" 필드를 추가하고 해당 "tableKey"를 "class\_scores"로 설정하여 실행에 의해 로그된 wandb.Table을 가져옵니다. 이를 통해 드롭다운 메뉴를 통해 wandb.Table의 열로 "class\_scores"로 로그된 두 히스토그램 구간 세트("red\_bins" 및 "blue\_bins")를 채울 수 있습니다. 제 예에서는 빨간 구간에는 "동물" 클래스 예측 점수, 파란 구간에는 "식물"을 선택했습니다.
4. 미리보기 렌더링에서 볼 수 있는 플롯에 만족할 때까지 Vega spec과 쿼리를 계속 변경합니다. 완료되면 상단의 "저장하기"를 클릭하여 커스텀 플롯에 이름을 지정하여 재사용할 수 있습니다. 그런 다음 "패널 라이브러리에서 적용하기"를 클릭하여 플롯을 완성하세요.

여기 제 실험에서의 결과가 나와 있습니다: 단 1000개의 예제로 한 에포크 동안 학습한 모델은 대부분의 이미지가 식물이 아니라고 매우 확신하며 어떤 이미지가 동물일 수 있는지에 대해 매우 불확실합니다.