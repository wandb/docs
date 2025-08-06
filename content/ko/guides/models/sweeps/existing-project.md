---
title: '튜토리얼: 프로젝트에서 스윕 작업 생성'
description: 기존 W&B 프로젝트에서 sweep job 을 생성하는 방법에 대한 튜토리얼입니다.
menu:
  default:
    identifier: ko-guides-models-sweeps-existing-project
    parent: sweeps
---

이 튜토리얼에서는 기존의 W&B 프로젝트에서 sweep job 을 생성하는 방법을 설명합니다. 우리는 [Fashion MNIST 데이터셋](https://github.com/zalandoresearch/fashion-mnist)을 사용하여 PyTorch 합성곱 신경망이 이미지를 분류하는 방법을 학습시킬 것입니다. 필요한 코드와 데이터셋은 [W&B examples repository (PyTorch CNN Fashion)](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)에 위치해 있습니다.

결과는 이 [W&B Dashboard](https://app.wandb.ai/carey/pytorch-cnn-fashion)에서 확인할 수 있습니다.

## 1. 프로젝트 생성하기

먼저, 베이스라인을 만듭니다. W&B examples GitHub 저장소에서 PyTorch MNIST 데이터셋 예제 모델을 다운로드합니다. 그리고 모델을 트레이닝합니다. 트레이닝 스크립트는 `examples/pytorch/pytorch-cnn-fashion` 디렉토리에 있습니다.

1. 이 저장소를 복제합니다: `git clone https://github.com/wandb/examples.git`
2. 예제 디렉토리로 이동합니다: `cd examples/pytorch/pytorch-cnn-fashion`
3. 트레이닝 스크립트를 직접 실행합니다: `python train.py`

원한다면, W&B App UI 대시보드에서 예제 결과를 확인해볼 수 있습니다.

[예제 프로젝트 페이지 보기 →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. 스윕 생성하기

프로젝트 페이지에서 사이드바의 [Sweep 탭]({{< relref path="./sweeps-ui.md" lang="ko" >}})을 열고 **Create Sweep**을 선택하세요.

{{< img src="/images/sweeps/sweep1.png" alt="Sweep overview" >}}

자동 생성된 설정은 지금까지 실행한 run 들을 기반으로 스윕할 값들을 예측하여 제안합니다. 원하는 하이퍼파라미터 범위로 값을 수정하고, 스윕을 시작하세요. 스윕이 실행되면 호스팅된 W&B 스윕 서버에서 새로운 프로세스가 시작됩니다. 이 중앙 서비스는 트레이닝 job 들을 실행하는 에이전트(agents)들을 조율합니다.

{{< img src="/images/sweeps/sweep2.png" alt="Sweep configuration" >}}

## 3. 에이전트 실행하기

이제 에이전트를 로컬에서 실행하세요. 여러 대의 머신에서 병렬로 최대 20개의 에이전트를 띄울 수 있으므로, 작업을 분산해 빠르게 스윕을 완료할 수 있습니다. 에이전트는 다음에 시도할 파라미터 세트를 출력해줍니다.

{{< img src="/images/sweeps/sweep3.png" alt="Launch agents" >}}

이제 스윕이 실행되고 있습니다. 다음 이미지는 예제 스윕 job 이 실행되는 동안 대시보드 모습입니다. [예제 프로젝트 페이지 보기 →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

{{< img src="/images/sweeps/sweep4.png" alt="Sweep dashboard" >}}

## 기존 run 들로 새로운 스윕 시작하기

이전에 로그한 기존 run 들을 이용해 새로운 스윕을 시작하세요.

1. 프로젝트 테이블을 엽니다.
2. 테이블 왼쪽에 있는 체크박스를 사용해 사용할 run 들을 선택합니다.
3. 드롭다운을 클릭해 새 스윕을 생성합니다.

이제 스윕이 서버에 세팅됩니다. 하나 이상의 에이전트를 실행하기만 하면 run 들이 자동으로 시작됩니다.

{{< img src="/images/sweeps/tutorial_sweep_runs.png" alt="Seed sweep from runs" >}}

{{% alert %}}
새 스윕을 bayesian sweep 으로 시작하면, 선택한 run 들이 Gaussian Process 의 시드로 사용됩니다.
{{% /alert %}}