---
title: Visualize and analyze tables
description: W&B Tables를 시각화하고 분석하세요.
menu:
  default:
    identifier: ko-guides-core-tables-visualize-tables
    parent: tables
weight: 2
---

기계 학습 모델의 성능에 대한 질문에 답변하고, 데이터를 분석하는 등 필요에 따라 W&B Tables를 사용자 정의하세요.

다음과 같은 방식으로 데이터를 상호 작용적으로 탐색할 수 있습니다:

* 모델, 에포크 또는 개별 예시 전반에서 변경 사항을 정확하게 비교
* 데이터에서 더 높은 수준의 패턴을 이해
* 시각적 샘플을 통해 인사이트를 캡처하고 전달

{{% alert %}}
W&B Tables는 다음과 같은 행동을 보입니다:
1. **아티팩트 컨텍스트에서 상태 비저장**: 아티팩트 버전과 함께 기록된 모든 테이블은 브라우저 창을 닫은 후 기본 상태로 재설정됩니다.
2. **워크스페이스 또는 리포트 컨텍스트에서 상태 저장**: 단일 run 워크스페이스, 다중 run 프로젝트 워크스페이스 또는 Report에서 테이블에 적용한 모든 변경 사항은 유지됩니다.

현재 W&B Table 뷰를 저장하는 방법에 대한 자세한 내용은 [뷰 저장]({{< relref path="#save-your-view" lang="ko" >}})을 참조하세요.
{{% /alert %}}

## 두 테이블을 보는 방법
[병합된 뷰]({{< relref path="#merged-view" lang="ko" >}}) 또는 [나란히 보기]({{< relref path="#side-by-side-view" lang="ko" >}})를 사용하여 두 테이블을 비교합니다. 예를 들어, 아래 이미지는 MNIST 데이터의 테이블 비교를 보여줍니다.

{{< img src="/images/data_vis/table_comparison.png" alt="Left: mistakes after 1 training epochs, Right: mistakes after 5 epochs" max-width="90%" >}}

다음 단계에 따라 두 테이블을 비교합니다:

1. W&B App에서 프로젝트로 이동합니다.
2. 왼쪽 패널에서 Artifacts 아이콘을 선택합니다.
3. 아티팩트 버전을 선택합니다.

다음 이미지에서는 5번의 에포크 이후 각 MNIST 검증 데이터에 대한 모델 예측을 보여줍니다 ([여기에서 대화형 예제 보기](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json)).

{{< img src="/images/data_vis/preds_mnist.png" alt="Click on 'predictions' to view the Table" max-width="90%" >}}

4. 사이드바에서 비교하려는 두 번째 아티팩트 버전 위로 마우스를 가져간 다음 나타나는 **비교**를 클릭합니다. 예를 들어, 아래 이미지에서는 5번의 트레이닝 에포크 후에 동일한 모델에서 수행한 MNIST 예측과 비교하기 위해 "v4"로 레이블이 지정된 버전을 선택합니다.

{{< img src="/images/data_vis/preds_2.png" alt="Preparing to compare model predictions after training for 1 epoch (v0, shown here) vs 5 epochs (v4)" max-width="90%" >}}

### 병합된 뷰

처음에는 두 테이블이 함께 병합되어 표시됩니다. 첫 번째로 선택한 테이블은 인덱스 0과 파란색 강조 표시가 있고 두 번째 테이블은 인덱스 1과 노란색 강조 표시가 있습니다. [여기에서 병합된 테이블의 라이브 예제를 봅니다](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec).

{{< img src="/images/data_vis/merged_view.png" alt="In the merged view, numerical columns appears as histograms by default" max-width="90%">}}

병합된 뷰에서 다음을 수행할 수 있습니다.

* **join 키 선택**: 왼쪽 상단의 드롭다운을 사용하여 두 테이블의 join 키로 사용할 열을 설정합니다. 일반적으로 이는 데이터셋의 특정 예제의 파일 이름 또는 생성된 샘플의 증가하는 인덱스와 같이 각 행의 고유 식별자입니다. 현재는 _모든_ 열을 선택할 수 있으며, 이로 인해 읽을 수 없는 테이블과 느린 쿼리가 발생할 수 있습니다.
* **join 대신 연결**: 이 드롭다운에서 "모든 테이블 연결"을 선택하여 열을 join하는 대신 두 테이블의 _모든 행을 합쳐서_ 하나의 더 큰 Table로 만듭니다.
* **각 Table을 명시적으로 참조**: 필터 표현식에서 0, 1 및 \*를 사용하여 하나 또는 두 테이블 인스턴스에서 열을 명시적으로 지정합니다.
* **자세한 숫자 차이를 히스토그램으로 시각화**: 모든 셀의 값을 한눈에 비교합니다.

### 나란히 보기

두 테이블을 나란히 보려면 첫 번째 드롭다운을 "테이블 병합: 테이블"에서 "목록: 테이블"로 변경한 다음 "페이지 크기"를 각각 업데이트합니다. 여기에서 첫 번째로 선택한 Table은 왼쪽에 있고 두 번째 Table은 오른쪽에 있습니다. 또한 "세로" 확인란을 클릭하여 이러한 테이블을 세로로 비교할 수도 있습니다.

{{< img src="/images/data_vis/side_by_side.png" alt="In the side-by-side view, Table rows are independent of each other." max-width="90%" >}}

* **테이블을 한눈에 비교**: 모든 작업 (정렬, 필터, 그룹)을 두 테이블에서 동시에 적용하고 변경 사항이나 차이점을 빠르게 찾습니다. 예를 들어, 추측별로 그룹화된 잘못된 예측, 전체적으로 가장 어려운 부정적 예측, 실제 레이블별 신뢰 점수 분포 등을 봅니다.
* **두 테이블을 독립적으로 탐색**: 스크롤하여 관심 있는 측면/행에 집중합니다.

## 아티팩트 비교
[시간 경과에 따른 테이블 비교]({{< relref path="#compare-tables-across-time" lang="ko" >}}) 또는 [모델 변형 비교]({{< relref path="#compare-tables-across-model-variants" lang="ko" >}})를 수행할 수도 있습니다.

### 시간 경과에 따른 테이블 비교
트레이닝의 의미 있는 각 단계에 대한 아티팩트에서 테이블을 기록하여 트레이닝 시간 경과에 따른 모델 성능을 분석합니다. 예를 들어, 모든 검증 단계가 끝날 때, 50번의 트레이닝 에포크마다 또는 파이프라인에 적합한 빈도로 테이블을 기록할 수 있습니다. 나란히 보기 기능을 사용하여 모델 예측의 변화를 시각화합니다.

{{< img src="/images/data_vis/compare_across_time.png" alt="For each label, the model makes fewer mistakes after 5 training epochs (R) than after 1 (L)" max-width="90%" >}}

트레이닝 시간 경과에 따른 예측 시각화에 대한 자세한 연습은 [이 리포트](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)와 이 대화형 [노트북 예제](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb?_gl=1*kf20ui*_gcl_au*OTI3ODM1OTcyLjE3MzE0MzU1NjU.*_ga*ODEyMjQ4MjkyLjE3MzE0MzU1NjU.*_ga_JH1SJHJQXJ*MTczMTcwNTMwNS45LjEuMTczMTcwNTM5My4zMy4wLjA.*_ga_GMYDGNGKDT*MTczMTcwNTMwNS44LjEuMTczMTcwNTM5My4wLjAuMA..)를 참조하세요.

### 모델 변형 간 테이블 비교

서로 다른 구성 (하이퍼파라미터, 기본 아키텍처 등)에서 모델 성능을 분석하기 위해 두 가지 다른 모델에 대해 동일한 단계에서 기록된 두 아티팩트 버전을 비교합니다.

예를 들어, 첫 번째 컨볼루션 레이어가 32에서 64로, 두 번째 레이어가 128에서 256으로 두 배가 되고 학습률이 0.001에서 0.002로 증가하는 `baseline`과 새로운 모델 변형 `2x_layers_2x_lr` 간의 예측을 비교합니다. [이 라이브 예제](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x_layers_2x_lr)에서 나란히 보기 기능을 사용하고 1회 (왼쪽 탭)와 5회 트레이닝 에포크 (오른쪽 탭) 후의 잘못된 예측으로 필터링합니다.

{{< tabpane text=true >}}
{{% tab header="1 training epoch" value="one_epoch" %}}
{{< img src="/images/data_vis/compare_across_variants.png" alt="After 1 epoch, performance is mixed: precision improves for some classes and worsens for others." >}}
{{% /tab %}}
{{% tab header="5 training epochs" value="five_epochs" %}}
{{< img src="/images/data_vis/compare_across_variants_after_5_epochs.png" alt="After 5 epochs, the 'double' variant is catching up to the baseline." >}}
{{% /tab %}}
{{< /tabpane >}}

## 뷰 저장

run 워크스페이스, 프로젝트 워크스페이스 또는 리포트에서 상호 작용하는 테이블은 뷰 상태를 자동으로 저장합니다. 테이블 작업을 적용한 다음 브라우저를 닫으면 다음에 테이블로 이동할 때 테이블이 마지막으로 본 구성을 유지합니다.

{{% alert %}}
아티팩트 컨텍스트에서 상호 작용하는 테이블은 상태를 유지하지 않습니다.
{{% /alert %}}

워크스페이스의 테이블을 특정 상태로 저장하려면 W&B Report로 내보냅니다. 테이블을 리포트로 내보내려면:
1. 워크스페이스 시각화 패널의 오른쪽 상단에 있는 케밥 아이콘 (세 개의 세로 점)을 선택합니다.
2. **패널 공유** 또는 **리포트에 추가**를 선택합니다.

{{< img src="/images/data_vis/share_your_view.png" alt="Share panel creates a new report, Add to report lets you append to an existing report." max-width="90%">}}

## 예제

다음 Reports는 W&B Tables의 다양한 유스 케이스를 강조합니다.

* [시간 경과에 따른 예측 시각화](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)
* [워크스페이스에서 테이블을 비교하는 방법](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)
* [이미지 및 분류 모델](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)
* [텍스트 및 생성 언어 모델](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)
* [개체명 인식](https://wandb.ai/stacey/ner_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)
* [AlphaFold 단백질](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)
