---
title: '튜토리얼: 커스텀 차트 사용하기'
description: W&B UI에서 커스텀 차트 기능을 사용하는 튜토리얼
menu:
  default:
    identifier: ko-guides-models-app-features-custom-charts-walkthrough
    parent: custom-charts
---

커스텀 차트를 사용하면 패널에 로드할 데이터와 시각화를 직접 제어할 수 있습니다.

## 1. 데이터 W&B에 로그하기

먼저, 여러분의 스크립트에서 데이터를 로그하세요. 트레이닝 시작 시 한 번만 지정하는 하이퍼파라미터와 같은 단일 값은 [wandb.Run.config]({{< relref path="/guides/models/track/config.md" lang="ko" >}})를 사용하세요. 여러 시점에 걸쳐 값을 기록하려면 [wandb.Run.log()]({{< relref path="/guides/models/track/log/" lang="ko" >}})를 사용하며, 커스텀 2D 배열은 `wandb.Table()`로 로그할 수 있습니다. 하나의 키에 10,000개 이하의 데이터 포인트를 기록하는 것을 추천합니다.

```python
with wandb.init() as run: 

  # 커스텀 데이터 테이블 로그하기
  my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
  run.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
  )
```

[간단한 예제 노트북](https://bit.ly/custom-charts-colab)에서 위처럼 데이터 테이블을 로그해볼 수 있습니다. 다음 단계에서는 커스텀 차트를 설정합니다. 최종 차트가 어떻게 보이는지는 [라이브 리포트](https://app.wandb.ai/demo-team/custom-charts/reports/Custom-Charts--VmlldzoyMTk5MDc)에서 확인할 수 있습니다.

## 2. 쿼리 생성하기

데이터를 시각화할 준비가 되었다면, 프로젝트 페이지로 이동해서 **`+`** 버튼을 클릭하여 새 패널을 추가하세요. 그 다음 **Custom Chart**를 선택하세요. [커스텀 차트 데모 워크스페이스](https://app.wandb.ai/demo-team/custom-charts)에서도 따라 해볼 수 있습니다.

{{< img src="/images/app_ui/create_a_query.png" alt="빈 커스텀 차트" >}}

### 쿼리 추가

1. `summary`를 클릭하고 `historyTable`을 선택하면 run 히스토리에서 데이터를 가져오는 새로운 쿼리를 설정할 수 있습니다.
2. `wandb.Table()`을 로그한 키를 입력하세요. 위 코드조각에서는 `my_custom_table`입니다. [예제 노트북](https://bit.ly/custom-charts-colab)에서는 키가 `pr_curve`와 `roc_curve`입니다.

### Vega 필드 지정하기

이제 쿼리가 해당 컬럼들을 불러왔으니, Vega 필드 드롭다운 메뉴에서 이 컬럼들을 선택할 수 있습니다:

{{< img src="/images/app_ui/set_vega_fields.png" alt="쿼리 결과에서 컬럼을 불러와 Vega 필드 설정" >}}

* **x축:** runSets_historyTable_r (리콜)
* **y축:** runSets_historyTable_p (프리시전)
* **색상:** runSets_historyTable_c (클래스 레이블)

## 3. 차트 커스터마이즈하기

지금 상태도 괜찮지만, 산점도가 아니라 라인 플롯으로 바꾸고 싶을 때도 있죠. **Edit** 버튼을 눌러 이 기본 차트의 Vega 스펙을 수정해보세요. [커스텀 차트 데모 워크스페이스](https://app.wandb.ai/demo-team/custom-charts)에서 실습 가능해요.

{{< img src="/images/general/custom-charts-1.png" alt="커스텀 차트 선택" >}}

저는 다음과 같이 Vega 스펙을 수정해서 시각화를 커스터마이즈했습니다:

* 플롯, 범례, x축, y축에 타이틀을 추가했습니다(각 항목의 “title” 설정)
* “mark” 값을 “point”에서 “line”으로 변경했습니다
* 사용하지 않는 “size” 필드를 제거했습니다

{{< img src="/images/app_ui/customize_vega_spec_for_pr_curve.png" alt="PR curve Vega spec" >}}

이 차트를 프로젝트 내에서 다른 곳에서도 쓸 수 있도록 프리셋으로 저장하려면, 페이지 상단의 **Save as** 버튼을 누르세요. 아래는 최종 결과와 ROC 커브 예시입니다:

{{< img src="/images/general/custom-charts-2.png" alt="PR curve 차트" >}}

## 보너스: 컴포지트 히스토그램

히스토그램은 수치형 분포를 시각화하여 더 큰 데이터셋을 이해하는 데 도움을 줍니다. 컴포지트 히스토그램은 같은 bin에 여러 분포를 보여주기 때문에, 서로 다른 모델이나 같은 모델 내에서 여러 클래스를 비교할 수 있습니다. 예를 들어, 운전 장면에서 오브젝트를 탐지하는 시멘틱 세그멘테이션 모델이라면 정확도 최적화와 IOU(intersection over union) 최적화의 효과를 비교하거나, 데이터에서 큰 비중을 차지하는 자동차(일반적이고 큼)와 교통 표지판(훨씬 작고 드문)의 탐지력을 비교할 수 있습니다. [데모 Colab](https://bit.ly/custom-charts-colab)에서는 10가지 생물 분류 중 두 클래스에 대한 confidence 스코어를 비교할 수 있습니다.

{{< img src="/images/app_ui/composite_histograms.png" alt="컴포지트 히스토그램" >}}

커스텀 컴포지트 히스토그램 패널을 직접 만들어보려면:

1. 워크스페이스나 Report에 새 Custom Chart 패널을 생성하세요(“Custom Chart” 시각화 추가). 오른쪽 상단의 “Edit” 버튼을 눌러 어떤 기본 패널 타입에서든 Vega 사양을 바로 수정할 수 있습니다.
2. 내 [컴포지트 히스토그램 Vega MVP 코드](https://gist.github.com/staceysv/9bed36a2c0c2a427365991403611ce21)를 복사해, 기존 Vega 사양을 교체하세요. 여러 부분(메인 타이틀, 축 타이틀, 입력 범위 등)을 [Vega 문법](https://vega.github.io/)을 통해 직접 수정할 수 있습니다. (색상을 바꾸거나, 세 번째 히스토그램도 추가할 수 있어요!)
3. 오른쪽 쿼리 창에서 wandb 로그 데이터 중 원하는 데이터를 불러오도록 수정하세요. `summaryTable` 필드를 추가하고, 해당 `tableKey`는 `class_scores`로 지정해 run에서 로그한 `wandb.Table`을 불러올 수 있습니다. 이제 `class_scores`로 기록된 wandb.Table의 컬럼들을 드롭다운에서 선택해 두 가지 히스토그램 bin 세트(`red_bins`, `blue_bins`)에 연결할 수 있습니다. 예를 들어, 저는 빨간색 bin에 ‘animal’ 클래스 예측값을, 파란색 bin에는 ‘plant’ 클래스를 선택했습니다.
4. 프리뷰 렌더에 원하는 플롯이 나올 때까지 Vega 사양과 쿼리를 계속 조정할 수 있습니다. 완성되면 페이지 상단의 **Save as**를 클릭, 커스텀 플롯의 이름을 지정해서 저장하세요. 그런 다음 **Apply from panel library**를 클릭해 플롯을 완성할 수 있습니다.

아래는 아주 간단한 실험 결과 예시입니다: 단 1,000개의 데이터로 1 에포크만 학습한 모델인데, 대부분의 이미지는 plant가 아니라고 확신하고, 어떤 이미지가 animal인지에 대해서는 불확실해하는 결과를 보여줍니다.

{{< img src="/images/general/custom-charts-3.png" alt="차트 설정" >}}

{{< img src="/images/general/custom-charts-4.png" alt="차트 결과" >}}