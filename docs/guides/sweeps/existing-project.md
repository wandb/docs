---
title: Tutorial: Create sweep job from project
description: 기존의 W&B 프로젝트에서 스윕 작업을 생성하는 방법에 대한 튜토리얼.
displayed_sidebar: default
---

이 튜토리얼은 기존의 W&B 프로젝트에서 스윕 작업을 생성하는 방법을 설명합니다. 우리는 PyTorch 합성곱 신경망을 사용하여 이미지를 분류하는 Fashion MNIST 데이터셋을 트레이닝할 것입니다. 필요한 코드와 데이터셋은 W&B 저장소에 있습니다: [https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)

이 [W&B 대시보드](https://app.wandb.ai/carey/pytorch-cnn-fashion)에서 결과를 탐색하세요.

## 1. 프로젝트 생성

먼저, 베이스라인을 생성합니다. W&B examples GitHub 저장소에서 PyTorch MNIST 데이터셋 예제 모델을 다운로드하세요. 그리고 모델을 트레이닝합니다. 트레이닝 스크립트는 `examples/pytorch/pytorch-cnn-fashion` 디렉토리에 있습니다.

1. 이 저장소를 복제합니다 `git clone https://github.com/wandb/examples.git`
2. 이 예제를 엽니다 `cd examples/pytorch/pytorch-cnn-fashion`
3. run을 수동으로 실행합니다 `python train.py`

선택적으로 W&B 앱 UI 대시보드에 나타나는 예제를 탐색하세요.

[예제 프로젝트 페이지 보기 →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. 스윕 생성

[프로젝트 페이지](../app/pages/project-page.md)에서 사이드바의 [스윕 탭](./sweeps-ui.md)을 열고 **Create Sweep**을 선택합니다.

![](/images/sweeps/sweep1.png)

자동 생성된 설정은 완료한 run을 기반으로 스윕할 파라미터 값을 예측합니다. 설정을 편집하여 시도할 하이퍼파라미터 범위를 지정하세요. 스윕을 시작하면, 호스팅된 W&B 스윕 서버에서 새로운 프로세스를 시작합니다. 이 중앙화된 서비스는 트레이닝 작업을 수행하는 에이전트들, 즉 머신들을 조정합니다.

![](/images/sweeps/sweep2.png)

## 3. 에이전트 실행

다음으로, 로컬에서 에이전트를 실행하세요. 작업을 분산시키고 스윕 작업을 더 빨리 완료하려면 서로 다른 머신에서 최대 20개의 에이전트를 병렬로 실행할 수 있습니다. 에이전트는 다음에 시도할 파라미터 세트를 출력할 것입니다.

![](/images/sweeps/sweep3.png)

이제 스윕이 실행되고 있습니다. 다음 이미지는 예제 스윕 작업이 실행됨에 따라 대시보드가 어떻게 보이는지를 보여줍니다. [예제 프로젝트 페이지 보기 →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

![](/images/sweeps/sweep4.png)

## 기존 run을 활용하여 새로운 스윕 시드하기

이전에 로그된 기존 run을 사용하여 새로운 스윕을 시작하세요.

1. 프로젝트 테이블을 엽니다.
2. 테이블의 왼쪽에 있는 체크박스를 사용하여 사용하려는 run을 선택합니다.
3. 드롭다운을 클릭하여 새로운 스윕을 생성합니다.

이제 우리의 서버에 스윕이 설정됩니다. 한 개 이상의 에이전트를 실행하여 run을 시작하기만 하면 됩니다.

![](/images/sweeps/tutorial_sweep_runs.png)

:::info
새로운 스윕을 베이지안 스윕으로 시작하면, 선택된 run이 Gaussian Process에도 시드됩니다.
:::