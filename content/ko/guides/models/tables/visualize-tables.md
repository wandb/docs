---
title: 테이블 시각화 및 분석
description: W&B Tables를 시각화하고 분석하세요.
menu:
  default:
    identifier: ko-guides-models-tables-visualize-tables
    parent: tables
weight: 2
---

W&B Tables를 활용해 기계학습 모델의 성능을 분석하고, 데이터 분석 및 다양한 질문에 대한 답을 얻을 수 있도록 테이블을 자유롭게 커스터마이즈할 수 있습니다.

데이터를 인터랙티브하게 탐색해보세요:

* 모델, 에포크 또는 개별 예시별 변화를 정확하게 비교
* 데이터에서 더 높은 수준의 패턴 파악
* 시각적 샘플로 인사이트 기록 및 공유

{{% alert %}}
W&B Tables의 행동은 아래와 같습니다:
1. **Artifacts 컨텍스트에서는 상태를 저장하지 않음**: 아티팩트 버전과 함께 기록된 테이블은 브라우저 창을 닫으면 기본 상태로 초기화됩니다.
2. **워크스페이스 또는 Report 컨텍스트에서는 상태를 저장함**: 단일 run 워크스페이스, 멀티-run 프로젝트 워크스페이스 또는 Report에서 테이블에 적용한 모든 변경 사항이 저장됩니다.

현재 W&B Table 뷰를 저장하는 방법은 [뷰 저장하기]({{< relref path="#save-your-view" lang="ko" >}})를 참고하세요.
{{% /alert %}}

## 두 개의 테이블 비교하기
[병합 뷰]({{< relref path="#merged-view" lang="ko" >}}) 또는 [좌우 비교 뷰]({{< relref path="#side-by-side-view" lang="ko" >}})로 두 개의 테이블을 비교할 수 있습니다. 아래 이미지는 MNIST 데이터의 테이블 비교 예시입니다.

{{< img src="/images/data_vis/table_comparison.png" alt="트레이닝 에포크 비교" max-width="90%" >}}

두 개의 테이블을 비교하려면 다음 단계를 따르세요:

1. W&B App에서 프로젝트로 이동합니다.
2. 왼쪽 패널에서 Artifacts 아이콘을 선택합니다.
3. 아티팩트 버전을 선택합니다.

다음 이미지는 다섯 번의 에포크 후, 각 에포크마다 MNIST 검증 데이터에 대한 모델 예측값을 나타냅니다 ([인터랙티브 예시 보기](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json)).

{{< img src="/images/data_vis/preds_mnist.png" alt="'predictions'를 클릭하여 Table 보기" max-width="90%" >}}

4. 사이드바에서 비교하고 싶은 두 번째 아티팩트 버전 위에 마우스를 올리고, **Compare** 버튼이 나타나면 클릭합니다. 예를 들어, 아래 이미지에서는 "v4"로 라벨링된 버전을 선택해 동일한 모델이 5번 트레이닝한 MNIST 예측값과 비교하고 있습니다.

{{< img src="/images/data_vis/preds_2.png" alt="모델 예측값 비교" max-width="90%" >}}

### 병합 뷰

처음에는 두 테이블이 병합되어 보여집니다. 가장 먼저 선택된 테이블은 인덱스 0과 파란색 강조, 두 번째 테이블은 인덱스 1과 노란색 강조가 표시됩니다. [병합 테이블의 라이브 예시 보기](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec).

{{< img src="/images/data_vis/merged_view.png" alt="병합 뷰" max-width="90%">}}

병합 뷰에서 다음과 같은 작업을 할 수 있습니다:

* **조인 키 선택**: 좌상단 드롭다운에서 두 테이블을 연결할 조인 키 컬럼을 지정하세요. 일반적으로 데이터셋의 예시 파일명이나 생성된 샘플의 인덱스 등 각 행의 고유 식별자가 사용됩니다. _모든_ 컬럼 선택이 가능하나, 해독이 어렵거나 느린 결과가 나올 수 있습니다.
* **조인 대신 이어붙이기**: 해당 드롭다운에서 "concatenating all tables"를 선택하면, 두 테이블의 모든 행을 하나의 큰 Table로 _합집합_ 할 수 있습니다 (컬럼 기준 조인 대신).
* **각 Table을 명시적으로 참조**: filter expression에서 0, 1, \*를 이용해 각 Table 인스턴스의 컬럼을 정확하게 지정할 수 있습니다.
* **숫자 값의 차이 시각화**: 셀별 값을 빠르게 비교할 수 있도록 히스토그램으로 시각화됩니다.

### 좌우 비교 뷰

두 테이블을 좌우에 나란히 살펴보려면, 첫 번째 드롭다운을 "Merge Tables: Table"에서 "List of: Table"로 변경하고 "Page size"도 원하는대로 조정하세요. 선택된 첫 번째 Table은 왼쪽, 두 번째 Table은 오른쪽에 표시됩니다. 또한 "Vertical" 체크박스를 클릭하면 테이블을 세로로 비교할 수도 있습니다.

{{< img src="/images/data_vis/side_by_side.png" alt="좌우 비교 테이블 뷰" max-width="90%" >}}

* **빠른 비교**: 정렬, 필터, 그룹 등의 작업을 동시에 적용해서 두 테이블의 차이를 한 번에 파악할 수 있습니다. 예를 들어, 예측이 잘못된 부분을 guess별로 그룹화하거나, 가장 구분이 어려운 음성 샘플, 라벨별 confidence 점수 분포 등을 볼 수 있습니다.
* **각 테이블을 개별적으로 탐색**: 관심 있는 부분의 행(또는 열)을 자유롭게 탐색할 수 있습니다.

## Run 전반에 걸친 값 변화 시각화

Table에 기록된 값들이 run 전체에서 어떻게 변화하는지 step 슬라이더로 볼 수 있습니다. 슬라이더를 움직이면 다양한 step에서 기록된 값들을 확인할 수 있습니다. 예를 들어, loss, accuracy 또는 기타 메트릭이 run 이후 어떻게 변화했는지 확인할 수 있습니다.

슬라이더는 값을 변화시키는 기준이 되는 키로 값을 결정합니다. 기본적으로 `_step` 키가 사용되며, 이는 W&B가 자동으로 기록해주는 특별한 키입니다. `_step` 키는 `wandb.Run.log()`가 호출될 때마다 1씩 증가하는 정수입니다.

W&B Table에 step 슬라이더를 추가하려면:

1. 프로젝트의 워크스페이스로 이동합니다.
2. 우측 상단에서 **Add panel**을 클릭합니다.
3. **Query panel**을 선택합니다.
4. 쿼리 표현식 에디터에서 `runs`를 선택하고 키보드의 **Enter**를 누릅니다.
5. 패널 설정을 보려면 톱니바퀴 아이콘을 클릭하세요.
6. **Render As** 선택기를 **Stepper**로 설정합니다.
7. **Stepper Key**를 `_step` 또는 [기준으로 사용할 키]({{< relref path="#custom-step-keys" lang="ko" >}})로 설정합니다.

아래 이미지는 세 개의 W&B run과 그들이 step 295에서 기록한 값들을 Query panel에서 보여주고 있습니다.

{{< img src="/images/data_vis/stepper_key.png" alt="Step 슬라이더 기능">}}

W&B App UI 내에서는 동일 값이 여러 step에서 중복 표시될 수 있습니다. 이는 여러 run이 서로 다른 step에서 동일한 값을 기록했거나, run이 모든 step에 값을 기록하지 않았을 때 발생할 수 있습니다. 특정 step에서 값이 누락된 경우, W&B는 마지막에 기록된 값을 슬라이더 키로 사용합니다.

### 사용자 정의 step 키

step 키는 run에서 기록하는 임의의 숫자형 메트릭이 될 수 있습니다 (예: `epoch` 또는 `global_step`). 사용자 정의 step 키를 사용하면, W&B가 해당 run에서 그 키의 각 값에 맞춰 step(`_step`)을 연결합니다.

아래 표는 사용자 정의 step 키 `epoch`가 세 개의 서로 다른 run (`serene-sponge`, `lively-frog`, `vague-cloud`)에서 `_step` 값에 어떻게 매핑되는지를 보여줍니다. 각 행은 특정 run에서 `_step` 값에 `wandb.Run.log()`가 호출된 사례를 나타냅니다. 열은 해당 step에서 기록된 epoch 값을 보여줍니다 (없는 값은 생략됨).

각 run에서 `wandb.Run.log()`가 최초 호출되었을 때는 `epoch` 값이 기록되지 않아, epoch 칸이 비어 있습니다.

| `_step` | vague-cloud (`epoch`) | lively-frog(`epoch`) |  serene-sponge (`epoch`) |
| ------- | ------------- | ----------- | ----------- |
| 1 | | |  |
| 2  |   |   | 1  | 
| 4  |   | 1 | 2  |
| 5  | 1 |   |  |
| 6  |  |   | 3  |
| 8  |  | 2 | 4  |
| 10 |  |   | 5  |
| 12 |  | 3 | 6  |
| 14 |  |   |  7 | 
| 15 | 2  |   |  |
| 16 |  | 4 | 8  | 
| 18 |  |   | 9  |
| 20 | 3 | 5 | 10 |

이제 슬라이더가 `epoch = 1`로 설정되면 다음과 같이 동작합니다:

* `vague-cloud`는 `epoch = 1`을 찾아 `_step = 5`에서 기록된 값을 반환
* `lively-frog`는 `epoch = 1`을 찾아 `_step = 4`에서 기록된 값을 반환
* `serene-sponge`는 `epoch = 1`을 찾아 `_step = 2`에서 기록된 값을 반환

슬라이더가 `epoch = 9`로 설정된 경우:

* `vague-cloud`는 `epoch = 9`가 없으므로, 직전의 `epoch = 3`을 사용해 `_step = 20`에서 기록된 값을 반환
* `lively-frog`는 `epoch = 9`가 없고, 직전 값인 `epoch = 5`를 찾아 `_step = 20`에서 기록된 값을 반환
* `serene-sponge`는 `epoch = 9`가 있으므로, `_step = 18`에서 기록된 값을 반환

## Artifacts 비교하기
[테이블을 시간에 따라 비교]({{< relref path="#compare-tables-across-time" lang="ko" >}})하거나, [모델 버전별로 비교]({{< relref path="#compare-tables-across-model-variants" lang="ko" >}})할 수도 있습니다.

### 시간에 따른 테이블 비교
트레이닝의 각 의미 있는 단계마다 Artifacts에 테이블을 기록하면 트레이닝 시간 전체에 걸친 모델 성능을 분석할 수 있습니다. 예를 들어 모든 검증 스텝 종료 시마다, 50 에포크마다, 또는 파이프라인에 적합한 빈도로 테이블을 기록할 수 있습니다. 좌우 비교 뷰로 모델 예측에서 어떻게 변화가 일어나는지 시각화해보세요.

{{< img src="/images/data_vis/compare_across_time.png" alt="트레이닝 진행 비교" max-width="90%" >}}

트레이닝 시간에 따른 예측값 시각화에 대한 자세한 예시는 [predictions over time report](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk) 및 이 인터랙티브 [notebook 예시](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb?_gl=1*kf20ui*_gcl_au*OTI3ODM1OTcyLjE3MzE0MzU1NjU.*_ga*ODEyMjQ4MjkyLjE3MzE0MzU1NjU.*_ga_JH1SJHJQXJ*MTczMTcwNTMwNS45LjEuMTczMTcwNTM5My4zMy4wLjA.*_ga_GMYDGNGKDT*MTczMTcwNTMwNS44LjEuMTczMTcwNTM5My4wLjAuMA..)를 참고하세요.

### 모델 버전 간 테이블 비교

같은 step에서 두 개의 서로 다른 모델에 대해 기록된 아티팩트 버전을 비교하여 다양한 설정(하이퍼파라미터, 기본 아키텍처 등)에서 모델 성능이 어떻게 다른지 분석할 수 있습니다.

예를 들어, `baseline` 모델과, 첫 번째 컨볼루션 레이어(32→64), 두 번째(128→256), 러닝레이트(0.001→0.002)가 두 배로 변한 새로운 모델 버전 `2x_layers_2x_lr`의 예측값을 비교할 수 있습니다. [이 라이브 예시](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x_layers_2x_lr)에서, 좌우 비교 뷰로 1 에포크(왼쪽 탭)와 5 에포크(오른쪽 탭) 이후 잘못된 예측값에 초점을 맞춰보세요.

{{< tabpane text=true >}}
{{% tab header="1 training epoch" value="one_epoch" %}}
{{< img src="/images/data_vis/compare_across_variants.png" alt="성능 비교" >}}
{{% /tab %}}
{{% tab header="5 training epochs" value="five_epochs" %}}
{{< img src="/images/data_vis/compare_across_variants_after_5_epochs.png" alt="모델 버전 성능 비교" >}}
{{% /tab %}}
{{< /tabpane >}}

## 뷰 저장하기

Run 워크스페이스, 프로젝트 워크스페이스, Report에서 Table을 다루면 자동으로 뷰 상태가 저장됩니다. 테이블에서 작업 후 브라우저를 닫았다가 다시 접속해도 마지막 상태가 그대로 유지됩니다.

{{% alert %}}
Artifacts 컨텍스트에서 다루는 Table은 상태가 저장되지 않습니다.
{{% /alert %}}

특정 상태의 Table을 워크스페이스에서 저장하고 싶다면 W&B Report로 내보낼 수 있습니다. Report로 내보내려면:
1. 워크스페이스 시각화 패널 우상단의 케밥 메뉴(세로 점 3개)를 클릭합니다.
2. **Share panel** 또는 **Add to report**를 선택합니다.

{{< img src="/images/data_vis/share_your_view.png" alt="Report 공유 옵션" max-width="90%">}}

## 예시

아래 Reports는 W&B Tables의 다양한 유스 케이스를 보여줍니다:

* [Visualize Predictions Over Time](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)
* [How to Compare Tables in Workspaces](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)
* [Image & Classification Models](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)
* [Text & Generative Language Models](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)
* [Named Entity Recognition](https://wandb.ai/stacey/ner_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)
* [AlphaFold Proteins](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)