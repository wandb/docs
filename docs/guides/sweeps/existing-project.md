---
description: Tutorial on how to create sweep jobs from a pre-existing W&B project.
displayed_sidebar: default
---

# 튜토리얼 - 기존 프로젝트에서 스윕 생성하기

<head>
    <title>기존 프로젝트에서 스윕 생성하기 튜토리얼</title>
</head>

이 튜토리얼은 기존 W&B 프로젝트에서 스윕 작업을 생성하는 방법을 단계별로 설명합니다. 이미지를 분류하는 방법을 학습하기 위해 PyTorch 합성곱 신경망을 사용하여 [Fashion MNIST 데이터세트](https://github.com/zalandoresearch/fashion-mnist)를 훈련시킬 것입니다. 필요한 코드와 데이터세트는 W&B 리포지토리에 있습니다: [https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)

이 [W&B 대시보드](https://app.wandb.ai/carey/pytorch-cnn-fashion)에서 결과를 탐색하세요.

## 1. 프로젝트 생성하기

먼저, 기준을 생성합니다. W&B 예제 GitHub 리포지토리에서 PyTorch MNIST 데이터세트 예제 모델을 다운로드합니다. 다음으로, 모델을 학습시킵니다. 학습 스크립트는 `examples/pytorch/pytorch-cnn-fashion` 디렉터리 내에 있습니다.

1. 이 리포지토리를 클론합니다 `git clone https://github.com/wandb/examples.git`
2. 이 예제를 엽니다 `cd examples/pytorch/pytorch-cnn-fashion`
3. 수동으로 실행을 실행합니다 `python train.py`

선택적으로 W&B 앱 UI 대시보드에서 예제를 탐색할 수 있습니다.

[예제 프로젝트 페이지 보기 →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. 스윕 생성하기

[프로젝트 페이지](../app/pages/project-page.md)에서 사이드바의 [스윕 탭](./sweeps-ui.md)을 열고 **스윕 생성**을 선택합니다.

![](@site/static/images/sweeps/sweep1.png)

자동 생성된 구성은 완료된 실행을 기반으로 스윕할 값들을 추측합니다. 구성을 편집하여 시도하고자 하는 하이퍼파라미터의 범위를 지정합니다. 스윕을 시작하면 호스팅된 W&B 스윕 서버에서 새 프로세스가 시작됩니다. 이 중앙 집중식 서비스는 에이전트를 조정합니다— 학습 작업을 실행하는 기계들입니다.

![](@site/static/images/sweeps/sweep2.png)

## 3. 에이전트 실행하기

다음으로, 로컬에서 에이전트를 실행합니다. 원한다면 작업을 분산시켜 스윕 작업을 더 빨리 마칠 수 있도록 다른 기계에서 최대 20개의 에이전트를 병렬로 실행할 수 있습니다. 에이전트는 다음에 시도할 파라미터 세트를 출력합니다.

![](@site/static/images/sweeps/sweep3.png)

이제 스윕을 실행하고 있습니다. 다음 이미지는 예제 스윕 작업이 실행되는 동안 대시보드가 어떻게 보이는지 보여줍니다. [예제 프로젝트 페이지 보기 →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

![](https://paper-attachments.dropbox.com/s\_5D8914551A6C0AABCD5718091305DD3B64FFBA192205DD7B3C90EC93F4002090\_1579066494222\_image.png)

## 기존 실행을 사용하여 새 스윕 시드하기

이전에 기록한 실행을 사용하여 새 스윕을 시작합니다.

1. 프로젝트 테이블을 엽니다.
2. 테이블 왼쪽의 체크박스를 사용하여 사용하고자 하는 실행을 선택합니다.
3. 드롭다운을 클릭하여 새 스윕을 생성합니다.

이제 스윕이 서버에 설정됩니다. 실행을 시작하기 위해 하나 이상의 에이전트를 실행하기만 하면 됩니다.

![](/images/sweeps/tutorial_sweep_runs.png)

:::info
새 스윕을 베이지안 스윕으로 시작하면 선택된 실행도 가우시안 프로세스를 시드합니다.
:::