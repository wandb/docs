---
title: Visualize and analyze tables
description: W&B Tables를 시각화하고 분석하세요.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

Customize your W&B Tables to answer questions about your machine learning model's performance, analyze your data, and more.

대화형으로 데이터를 탐색하여:

* 모델, 에포크, 또는 개별 예제 간의 변화를 정확하게 비교합니다
* 데이터의 고차원 패턴을 이해합니다
* 시각적 샘플을 통해 통찰력을 캡처하고 전달합니다

:::info
W&B Tables는 다음과 같은 행동을 가집니다:
1. **아티팩트 컨텍스트에서의 무상태성**: 아티팩트 버전과 함께 기록된 테이블은 브라우저 창을 닫으면 기본 상태로 리셋됩니다.
2. **워크스페이스나 리포트 컨텍스트에서의 상태 보존성**: 단일 run 워크스페이스, 다중 run 프로젝트 워크스페이스, 또는 리포트에서 테이블에 수행한 모든 변경 사항은 유지됩니다.

현재 W&B Table 뷰를 저장하는 방법에 대한 정보는 [Save your view](#save-your-view)를 참조하세요.
:::

## How to view two tables
두 테이블을 [병합된 뷰](#merged-view) 또는 [나란한 뷰](#side-by-side-view)로 비교하세요. 예를 들어, 아래 이미지는 MNIST 데이터의 테이블 비교를 보여줍니다.

![왼쪽: 1 에포크 트레이닝 후 실수, 오른쪽: 5 에포크 후 실수](/images/data_vis/table_comparison.png)

두 테이블을 비교하려면 다음 단계를 따르세요:

1. W&B 앱에서 당신의 프로젝트로 이동합니다.
2. 왼쪽 패널에서 Artifacts 아이콘을 선택하세요.
3. 아티팩트 버전을 선택합니다.

아래 이미지는 5 에포크 후 MNIST 검증 데이터에 대한 모델의 예측을 보여줍니다 ([여기서 대화형 예제 보기](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json)).

![테이블을 보려면 "predictions"를 클릭하세요](/images/data_vis/preds_mnist.png)

4. 비교하고자 하는 두 번째 아티팩트 버전 위에 사이드바로 마우스를 올리면 **Compare**이 나타날 때 클릭합니다. 예를 들어, 아래 이미지는 동일한 모델이 5 에포크 트레이닝 후 MNIST 예측과 비교하기 위해 "v4"로 레이블된 버전을 선택하는 것을 보여줍니다.

![1 에포크(v0, 여기 표시됨) 와 5 에포크(v4) 트레이닝 후 모델 예측 비교 준비 중](/images/data_vis/preds_2.png)

### Merged view

처음에는 두 테이블이 병합된 상태로 나타납니다. 첫 번째로 선택된 테이블은 인덱스 0과 파란색 하이라이트를 가지며, 두 번째 테이블은 인덱스 1과 노란색 하이라이트를 가집니다. [병합된 테이블의 라이브 예제 보기](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec).

![병합된 뷰에서, 숫자형 컬럼들은 기본적으로 히스토그램으로 나타납니다](/images/data_vis/merged_view.png)

병합된 뷰에서, 당신은

* **조인 키를 선택할 수 있습니다**: 왼쪽 상단에서 드롭다운을 사용하여 두 테이블의 조인 키로 사용할 열을 설정하세요. 일반적으로 이는 데이터셋의 특정 예제의 파일 이름이나 생성된 샘플의 증가하는 인덱스와 같은 각 행의 고유 식별자일 것입니다. 현재는 _어떤_ 열도 선택할 수 있으며, 이는 읽기 어려운 테이블을 만들고 쿼리를 느리게 할 수 있습니다.
* **조인 대신 연결할 수 있습니다**: 이 드롭다운에서 "모든 테이블 연결"을 선택하여 두 테이블의 모든 행을 열 간의 조인 대신에 하나의 더 큰 테이블로 _통합_ 할 수 있습니다.
* **각 테이블을 명시적으로 참조할 수 있습니다**: 필터 표현식에서 0, 1, \*을 사용하여 하나 또는 두 테이블 인스턴스 중 특정 열을 명확하게 지정할 수 있습니다.
* **자세한 숫자 차이를 히스토그램으로 시각화할 수 있습니다**: 셀의 값들을 한눈에 비교하세요.

### Side-by-side view

두 테이블을 나란히 보려면 첫 번째 드롭다운을 "Merge Tables: Table"에서 "List of: Table"로 변경하고 "Page size"를 각각 업데이트하세요. 여기서 첫 번째로 선택된 테이블은 왼쪽에 있고, 두 번째 테이블은 오른쪽에 있습니다. 또한, "Vertical" 체크박스를 클릭하여 이 테이블을 수직으로 비교할 수도 있습니다.

![나란한 뷰에서는, 테이블 행들이 서로 독립적입니다.](/images/data_vis/side_by_side.png)

* **테이블을 한눈에 비교할 수 있습니다**: 두 테이블에 동시에 정렬, 필터, 그룹화 등의 작업을 적용하여 변동이나 차이를 빠르게 확인합니다. 예를 들어, 추측에 따라 그룹화된 틀린 예측을 보거나, 전체적으로 가장 어려운 음성 예측, 실제 레이블에 따른 신뢰도 점수 분포 등을 볼 수 있습니다.
* **두 테이블을 독립적으로 탐색할 수 있습니다**: 관심 있는 부분/행을 스크롤하고 집중하세요.

## Compare artifacts
또한 [시간 경과에 따른 테이블 비교](#compare-tables-across-time) 또는 [모델 변형 비교](#compare-tables-across-model-variants)을 수행할 수 있습니다.

### Compare tables across time
모델 성능을 트레이닝 시간이 지남에 따라 분석하기 위해 각각의 의미 있는 트레이닝 단계에서 아티팩트에 테이블을 기록하세요. 예를 들어, 모든 검증 단계 끝에, 50 에포크 트레이닝 후마다, 또는 당신의 파이프라인에 맞는 주기마다 테이블을 기록할 수 있습니다. 모델 예측의 변화를 시각화하기 위해 나란한 뷰를 사용하세요.

![각 레이블에 대해, 5 에포크 트레이닝 후(오른쪽), 1 에포크 후(왼쪽)보다 모델이 더 적은 실수를 합니다.](/images/data_vis/compare_across_time.png)

트레이닝 시간에 따른 예측 시각화를 보다 자세히 살펴보려면, [이 리포트](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)와 이 대화형 [노트북 예제](http://wandb.me/tables-walkthrough)를 참조하세요.

### Compare tables across model variants

두 가지 다른 모델에 대해 동일한 단계에서 기록된 두 아티팩트 버전을 비교하여 설정(하이퍼파라미터, 기본 아키텍처 등)이 다른 모델의 성능을 분석합니다.

예를 들어, `baseline`과 새로운 모델 변형 `2x_layers_2x_lr` 사이의 예측을 비교합니다. 여기서 첫 번째 컨볼루션 층은 32에서 64로, 두 번째는 128에서 256으로, 학습률은 0.001에서 0.002로 증가합니다. [이 라이브 예제](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x_layers_2x_lr)를 사용하여 나란한 뷰로 1 에포크(왼쪽 탭)와 5 에포크(오른쪽 탭) 트레이닝 후의 틀린 예측을 필터링하세요.

<Tabs
  defaultValue="one_epoch"
  values={[
    {label: '1 training epoch', value: 'one_epoch'},
    {label: '5 training epochs', value: 'five_epochs'},
  ]}>
  <TabItem value="one_epoch">

![1 에포크 후, 성능은 혼합되어 일부 클래스에서는 정밀도가 개선되고 다른 클래스에서는 악화됩니다.](/images/data_vis/compare_across_variants.png)
  </TabItem>
  <TabItem value="five_epochs">

![5 에포크 후, "double" 변형은 베이스라인에 근접해가고 있습니다.](/images/data_vis/compare_across_variants_after_5_epochs.png)
  </TabItem>
</Tabs>

## Save your view

run workspace, project workspace 또는 리포트에서 상호작용하는 테이블은 자동으로 뷰 상태를 저장합니다. 테이블 작업을 적용한 후 브라우저를 닫아도, 테이블은 다음에 테이블로 탐색할 때 마지막으로 본 설정을 유지합니다.

:::tip
아티팩트 컨텍스트에서 상호작용하는 테이블은 무상태로 유지됩니다.
:::

워크스페이스에서 특정 상태로 테이블을 저장하려면, 그것을 W&B 리포트에 내보내세요. 리포트로 테이블을 내보내려면:
1. 워크스페이스 시각화 패널의 오른쪽 상단 모서리에 있는 케밥 아이콘(세 개의 세로 점)을 선택하세요.
2. **Share panel** 또는 **Add to report**를 선택하세요.

![Share panel은 새 리포트를 생성하고, Add to report는 기존 리포트에 추가합니다.](/images/data_vis/share_your_view.png)

## Examples

다음 리포트들은 W&B Tables의 다양한 유스 케이스를 강조합니다:

* [Visualize Predictions Over Time](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)
* [How to Compare Tables in Workspaces](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)
* [Image & Classification Models](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)
* [Text & Generative Language Models](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)
* [Named Entity Recognition](https://wandb.ai/stacey/ner_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)
* [AlphaFold Proteins](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)