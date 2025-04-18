---
title: 'Tutorial: Create sweep job from project'
description: 기존 W&B 프로젝트에서 스윕 작업을 생성하는 방법에 대한 튜토리얼입니다.
menu:
  default:
    identifier: ko-guides-models-sweeps-existing-project
    parent: sweeps
---

이 튜토리얼에서는 기존의 W&B 프로젝트에서 스윕 작업을 생성하는 방법을 설명합니다. [Fashion MNIST 데이터셋](https://github.com/zalandoresearch/fashion-mnist)을 사용하여 이미지를 분류하는 방법을 PyTorch 컨볼루션 신경망을 트레이닝합니다. 필요한 코드와 데이터셋은 W&B 저장소에 있습니다: [https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)

이 [W&B 대시보드](https://app.wandb.ai/carey/pytorch-cnn-fashion)에서 결과를 살펴보세요.

## 1. 프로젝트 생성

먼저, 베이스라인을 만듭니다. W&B 예제 GitHub 저장소에서 PyTorch MNIST 데이터셋 예제 모델을 다운로드합니다. 다음으로, 모델을 트레이닝합니다. 트레이닝 스크립트는 `examples/pytorch/pytorch-cnn-fashion` 디렉토리 내에 있습니다.

1. 이 저장소를 클론합니다: `git clone https://github.com/wandb/examples.git`
2. 이 예제를 엽니다: `cd examples/pytorch/pytorch-cnn-fashion`
3. run을 수동으로 실행합니다: `python train.py`

선택적으로 W&B App UI 대시보드에 나타나는 예제를 탐색합니다.

[예제 프로젝트 페이지 보기 →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. 스윕 생성

프로젝트 페이지에서 사이드바의 [Sweep tab]({{< relref path="./sweeps-ui.md" lang="ko" >}})을 열고 **Create Sweep**을 선택합니다.

{{< img src="/images/sweeps/sweep1.png" alt="" >}}

자동 생성된 설정은 완료한 run을 기반으로 스윕할 값을 추측합니다. 시도할 하이퍼파라미터 범위를 지정하도록 설정을 편집합니다. 스윕을 시작하면 호스팅된 W&B 스윕 서버에서 새 프로세스가 시작됩니다. 이 중앙 집중식 서비스는 트레이닝 작업을 실행하는 머신인 에이전트를 조정합니다.

{{< img src="/images/sweeps/sweep2.png" alt="" >}}

## 3. 에이전트 시작

다음으로, 로컬에서 에이전트를 시작합니다. 작업을 분산하고 스윕 작업을 더 빨리 완료하려면 최대 20개의 에이전트를 서로 다른 머신에서 병렬로 시작할 수 있습니다. 에이전트는 다음에 시도할 파라미터 세트를 출력합니다.

{{< img src="/images/sweeps/sweep3.png" alt="" >}}

이제 스윕을 실행하고 있습니다. 다음 이미지는 예제 스윕 작업이 실행되는 동안 대시보드가 어떻게 보이는지 보여줍니다. [예제 프로젝트 페이지 보기 →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

{{< img src="/images/sweeps/sweep4.png" alt="" >}}

## 기존 run으로 새 스윕 시드하기

이전에 기록한 기존 run을 사용하여 새 스윕을 시작합니다.

1. 프로젝트 테이블을 엽니다.
2. 테이블 왼쪽에서 확인란을 사용하여 사용할 run을 선택합니다.
3. 드롭다운을 클릭하여 새 스윕을 만듭니다.

이제 스윕이 서버에 설정됩니다. run 실행을 시작하려면 하나 이상의 에이전트를 시작하기만 하면 됩니다.

{{< img src="/images/sweeps/tutorial_sweep_runs.png" alt="" >}}

{{% alert %}}
새 스윕을 베이지안 스윕으로 시작하면 선택한 run도 가우스 프로세스를 시드합니다.
{{% /alert %}}
