---
description: Tutorial on how to create sweep jobs from a pre-existing W&B project.
displayed_sidebar: default
---

# 튜토리얼 - 기존 프로젝트에서 스윕 생성하기

<head>
    <title>기존 프로젝트에서 스윕 생성하기 튜토리얼</title>
</head>

이 튜토리얼은 기존 W&B 프로젝트에서 스윕 작업을 생성하는 방법에 대한 단계를 안내합니다. 이미지를 분류하는 방법을 학습하기 위해 PyTorch 컨볼루션 신경망을 훈련시키는 데 [Fashion MNIST 데이터셋](https://github.com/zalandoresearch/fashion-mnist)을 사용할 것입니다. 필요한 코드와 데이터셋은 W&B 리포지토리에 위치해 있습니다: [https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)

이 [W&B 대시보드](https://app.wandb.ai/carey/pytorch-cnn-fashion)에서 결과를 탐색하세요.

## 1. 프로젝트 생성하기

먼저, 베이스라인을 생성하세요. W&B 예제 GitHub 리포지토리에서 PyTorch MNIST 데이터셋 예제 모델을 다운로드합니다. 다음, 모델을 훈련하세요. 트레이닝 스크립트는 `examples/pytorch/pytorch-cnn-fashion` 디렉토리 안에 있습니다.

1. 이 리포지토리를 클론하세요 `git clone https://github.com/wandb/examples.git`
2. 이 예제를 열기 `cd examples/pytorch/pytorch-cnn-fashion`
3. 수동으로 run 실행하기 `python train.py`

선택적으로 W&B App UI 대시보드에 나타나는 예제를 탐색하세요.

[예제 프로젝트 페이지 보기 →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. 스윕 생성하기

[프로젝트 페이지](../app/pages/project-page.md)에서 사이드바의 [Sweep 탭](./sweeps-ui.md)을 열고 **Create Sweep**을 선택하세요.

![](@site/static/images/sweeps/sweep1.png)

자동 생성된 설정은 완료된 run을 기반으로 스윕할 값들을 추정합니다. 설정을 편집하여 시도하고 싶은 하이퍼파라미터의 범위를 지정하세요. 스윕을 시작하면 호스티드 W&B 스윕 서버에서 새 프로세스가 시작됩니다. 이 중앙 집중식 서비스는 트레이닝 작업을 실행하는 기계인 에이전트를 조정합니다.

![](@site/static/images/sweeps/sweep2.png)

## 3. 에이전트 실행하기

다음으로, 로컬에서 에이전트를 실행하세요. 작업을 분산시키고 스윕 작업을 더 빨리 완료하려면 최대 20개의 에이전트를 다른 기계에서 병렬로 실행할 수 있습니다. 에이전트는 다음에 시도할 파라미터 세트를 출력합니다.

![](@site/static/images/sweeps/sweep3.png)

이제 스윕을 실행 중입니다. 다음 이미지는 예제 스윕 작업이 실행되는 동안 대시보드가 어떻게 보이는지 보여줍니다. [예제 프로젝트 페이지 보기 →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

![](https://paper-attachments.dropbox.com/s\_5D8914551A6C0AABCD5718091305DD3B64FFBA192205DD7B3C90EC93F4002090\_1579066494222\_image.png)

## 기존 run으로 새 스윕 시작하기

이전에 기록한 기존 run을 사용하여 새 스윕을 시작하세요.

1. 프로젝트 테이블을 엽니다.
2. 테이블 왼쪽에 있는 체크박스를 선택하여 사용하려는 run을 선택합니다.
3. 새 스윕을 생성하기 위해 드롭다운을 클릭합니다.

이제 스윕이 서버에서 설정됩니다. 시작하려면 하나 이상의 에이전트를 실행하기만 하면 됩니다.

![](/images/sweeps/tutorial_sweep_runs.png)

:::info
새 스윕을 베이지안 스윕으로 시작하면, 선택한 run도 가우시안 프로세스를 시딩할 것입니다.
:::