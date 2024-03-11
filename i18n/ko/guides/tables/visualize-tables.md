---
description: Visualize and analyze W&B Tables.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 테이블 시각화 및 분석

W&B 테이블을 커스터마이즈하여 기계학습 모델의 성능에 대한 질문에 답하고, 데이터를 분석하고, 그 이상의 작업을 수행하세요.

데이터를 상호작용적으로 탐색하여:

* 모델, 에포크 또는 개별 예제 간의 변경사항을 정확하게 비교
* 데이터에서 고급 패턴을 이해
* 시각적 샘플로 인사이트를 포착하고 전달

:::info
W&B 테이블은 다음과 같은 행동을 가지고 있습니다:
1. **아티팩트 컨텍스트에서 상태 없음**: 아티팩트 버전과 함께 기록된 테이블은 브라우저 창을 닫은 후 기본 상태로 재설정됩니다.
2. **워크스페이스 또는 리포트 컨텍스트에서 상태 있음**: 단일 run 워크스페이스, 다중 run 프로젝트 워크스페이스 또는 리포트에서 테이블에 대한 변경사항은 유지됩니다.

현재 W&B 테이블 뷰를 저장하는 방법에 대한 정보는 [뷰 저장](#save-your-view)을 참조하세요.
:::

## 두 테이블 보기
[병합 뷰](#merged-view) 또는 [나란히 보기](#side-by-side-view)를 사용하여 두 테이블을 비교하세요. 예를 들어, 아래 이미지는 MNIST 데이터의 테이블 비교를 보여줍니다.

![왼쪽: 1 에포크 트레이닝 후의 실수, 오른쪽: 5 에포크 후의 실수](/images/data_vis/table_comparison.png)

두 테이블을 비교하려면 다음 단계를 따르세요:

1. W&B 앱에서 프로젝트로 이동하세요.
2. 왼쪽 패널에서 아티팩트 아이콘을 선택하세요.
2. 아티팩트 버전을 선택하세요.

다음 이미지에서는 5개 에포크 간격으로 MNIST 검증 데이터에 대한 모델의 예측값을 보여줍니다([여기에서 인터랙티브 예제 보기](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json)).

![테이블 보기를 위해 "예측값" 클릭](@site/static/images/data_vis/preds_mnist.png)

3. 비교하려는 두 번째 아티팩트 버전을 사이드바에서 마우스로 가리키고 **비교**가 나타나면 클릭하세요. 예를 들어, 아래 이미지에서는 5 에포크의 트레이닝 후 동일한 모델에 의해 만들어진 MNIST 예측값과 비교하기 위해 "v4"로 표시된 버전을 선택합니다.

![1 에포크 트레이닝 후 모델 예측값 비교 준비 (v0, 여기에 표시됨) vs 5 에포크 (v4)](@site/static/images/data_vis/preds_2.png)

### 병합 뷰

초기에 두 테이블이 함께 병합되어 표시됩니다. 첫 번째로 선택된 테이블은 인덱스 0과 파란색 하이라이트를 가지고, 두 번째 테이블은 인덱스 1과 노란색 하이라이트를 가집니다. [여기에서 병합된 테이블의 라이브 예제를 봅니다](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec).

![병합 뷰에서, 숫자 열은 기본적으로 히스토그램으로 표시됩니다](@site/static/images/data_vis/merged_view.png)

병합 뷰에서, 당신은

* **조인 키 선택**: 왼쪽 상단의 드롭다운을 사용하여 두 테이블을 조인하기 위한 열을 설정합니다. 일반적으로 이것은 데이터셋의 특정 예제 파일 이름이나 생성된 샘플에 대한 증가 인덱스와 같은 각 행의 고유 식별자가 됩니다. 현재 _모든_ 열을 선택할 수 있지만, 이는 이해하기 어려운 테이블과 느린 쿼리를 초래할 수 있습니다.
* **조인 대신 연결**: 이 드롭다운에서 "모든 테이블 연결"을 선택하여 두 테이블의 모든 행을 하나의 더 큰 테이블로 _합치는 대신_ 열 간에 조인합니다.
* **각 테이블을 명시적으로 참조**: 필터 표현식에서 0, 1, 및 *를 사용하여 한 테이블 또는 두 테이블 인스턴스의 열을 명시적으로 지정합니다.
* **히스토그램으로 상세한 숫자 차이 시각화**: 한눈에 어떤 셀의 값을 비교합니다.

### 나란히 보기

두 테이블을 나란히 보려면 첫 번째 드롭다운을 "테이블 병합: 테이블"에서 "테이블 목록: 테이블"로 변경한 다음 "페이지 크기"를 각각 업데이트하세요. 여기서 첫 번째로 선택된 테이블은 왼쪽에 있고 두 번째 테이블은 오른쪽에 있습니다. 또한 "세로" 체크박스를 클릭하여 이 테이블들을 세로로도 비교할 수 있습니다.

![나란히 보기에서, 테이블 행은 서로 독립적입니다.](/images/data_vis/side_by_side.png)

* **한눈에 테이블 비교**: 두 테이블에 동시에 모든 작업(정렬, 필터, 그룹)을 적용하고 빠르게 변경사항이나 차이를 파악하세요. 예를 들어, 추측별로 그룹화된 잘못된 예측값, 전반적으로 가장 어려운 부정, 진실 레이블별 신뢰 점수 분포 등을 보세요.
* **두 테이블을 독립적으로 탐색**: 스크롤하여 관심 있는 측면/행에 집중하세요

## 아티팩트 비교
[시간에 걸쳐 테이블 비교](#compare-across-time) 또는 [모델 변형 비교](#compare-across-model-variants)도 할 수 있습니다.

### 시간에 걸쳐 테이블 비교
트레이닝 시간에 따른 모델 성능을 분석하기 위해 트레이닝의 각 의미 있는 단계마다 아티팩트에 테이블을 기록하세요. 예를 들어, 트레이닝의 모든 50 에포크 후나 검증 단계의 끝에 테이블을 기록하거나 파이프라인에 맞는 빈도로 기록할 수 있습니다. 나란히 보기를 사용하여 모델 예측값의 변화를 시각화합니다.

![각 레이블에 대해 모델은 5 에포크 트레이닝 후(R) 1 에포크 후(L)보다 실수가 적습니다.](/images/data_vis/compare_across_time.png)

트레이닝 시간에 걸쳐 예측값을 시각화하는 보다 상세한 워크스루는 [이 리포트](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)와 이 인터랙티브 [노트북 예제](http://wandb.me/tables-walkthrough)를 참조하세요.

### 모델 변형 비교

두 가지 다른 설정(하이퍼파라미터, 기본 아키텍처 등)의 모델에 대해 동일한 단계에서 기록된 두 아티팩트 버전을 비교하여 모델 성능을 분석하세요.

예를 들어, `baseline`과 새로운 모델 변형 `2x_layers_2x_lr` 사이의 예측값을 비교하세요. 여기서 첫 번째 컨볼루션 레이어는 32에서 64로, 두 번째는 128에서 256으로 두 배가 되고, 학습률은 0.001에서 0.002로 두 배가 됩니다. [이 라이브 예제](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x\_layers\_2x\_lr)에서 나란히 보기를 사용하고 1 에포크(왼쪽 탭) 후의 잘못된 예측값과 5 에포크(오른쪽 탭) 후의 것을 필터링합니다.

<Tabs
  defaultValue="one_epoch"
  values={[
    {label: '1 에포크 트레이닝', value: 'one_epoch'},
    {label: '5 에포크 트레이닝', value: 'five_epochs'},
  ]}>
  <TabItem value="one_epoch">

![1 에포크 후, 성능은 혼합됩니다: 일부 클래스에 대한 정밀도는 개선되고 일부는 악화됩니다.](/images/data_vis/compare_across_variants.png)
  </TabItem>
  <TabItem value="five_epochs">

![5 에포크 후, "double" 변형이 베이스라인을 따라잡고 있습니다.](/images/data_vis/compare_across_variants_after_5_epochs.png)
  </TabItem>
</Tabs>

## 뷰 저장

run 워크스페이스, 프로젝트 워크스페이스 또는 리포트에서 상호작용하는 테이블은 자동으로 뷰 상태를 저장합니다. 테이블 작업을 적용한 후 브라우저를 닫으면, 다음에 테이블로 다시 이동할 때 테이블은 마지막으로 본 구성을 유지합니다.

:::tip
아티팩트 컨텍스트에서 상호작용하는 테이블은 상태 없이 유지됩니다.
:::

특정 상태에서 워크스페이스의 테이블을 저장하려면, W&B 리포트로 내보내세요. 리포트로 테이블을 내보내려면:
1. 워크스페이스 시각화 패널 오른쪽 상단에 있는 케밥 아이콘(세로로 세 개의 점)을 선택하세요.
2. **패널 공유** 또는 **리포트에 추가**를 선택하세요.

![패널 공유는 새 리포트를 생성하고, 리포트에 추가는 기존 리포트에 추가하는 것을 허용합니다.](/images/data_vis/share_your_view.png)

## 예제

이 리포트들은 W&B 테이블의 다양한 유스 케이스를 강조합니다:

* [시간에 걸쳐 예측값 시각화](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)
* [워크스페이스에서 테이블 비교 방법](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)
* [이미지 및 분류 모델](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)
* [텍스트 및 생성 언어 모델](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)
* [명명된 엔티티 인식](https://wandb.ai/stacey/ner\_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)
* [AlphaFold 단백질](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)