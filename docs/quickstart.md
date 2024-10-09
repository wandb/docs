---
title: W&B Quickstart
description: W&B 퀵스타트
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

Install W&B하고 몇 분 안에 기계학습 실험을 추적하기 시작하세요.

## 1. 계정 생성 및 W&B 설치하기
시작하기 전에 계정을 생성하고 W&B를 설치하세요:

1. [여기](https://wandb.ai/site)에서 무료 계정을 [등록](https://wandb.ai/site)한 후, wandb 계정에 로그인하세요.  
2. Python 3 환경에서 [`pip`](https://pypi.org/project/wandb/)을 사용하여 로컬 머신에 wandb 라이브러리를 설치하세요.  

다음 코드조각은 W&B CLI 및 Python 라이브러리를 사용하여 W&B를 설치하고 로그인하는 방법을 보여줍니다:

<Tabs
  defaultValue="notebook"
  values={[
    {label: 'Notebook', value: 'notebook'},
    {label: 'Command Line', value: 'cli'},
  ]}>
  <TabItem value="cli">

Weights and Biases API와 상호작용하기 위해 CLI 및 Python 라이브러리를 설치하세요:

```bash
pip install wandb
```

  </TabItem>
  <TabItem value="notebook">

Weights and Biases API와 상호작용하기 위해 CLI 및 Python 라이브러리를 설치하세요:

```notebook
!pip install wandb
```

  </TabItem>
</Tabs>

## 2. W&B에 로그인하기

<Tabs
  defaultValue="notebook"
  values={[
    {label: 'Notebook', value: 'notebook'},
    {label: 'Command Line', value: 'cli'},
  ]}>
  <TabItem value="cli">

다음으로, W&B에 로그인하세요:

```bash
wandb login
```

만약 [W&B 서버](./guides/hosting) (포함 **전용 클라우드** 또는 **자체 관리**)를 사용 중이라면: 

```bash
wandb login --relogin --host=http://your-shared-local-host.com
```

필요한 경우, 배포 관리자에게 호스트명을 요청하세요.

프롬프트가 표시되면 [API 키](https://wandb.ai/authorize)를 제공하세요.

  </TabItem>
  <TabItem value="notebook">

다음으로, W&B Python SDK를 가져오고 로그인하세요:

```python
wandb.login()
```

프롬프트가 표시되면 [API 키](https://wandb.ai/authorize)를 제공하세요.
  </TabItem>
</Tabs>

## 3. Run 시작하고 하이퍼파라미터 추적하기

[`wandb.init()`](./ref/python/run.md)를 사용하여 Python 스크립트나 노트북에서 W&B Run 오브젝트를 초기화하고 하이퍼파라미터 이름과 값을 key-value 쌍으로 `config` 파라미터에 딕셔너리로 전달하세요:

```python
run = wandb.init(
    # 이 run이 기록될 프로젝트 설정
    project="my-awesome-project",
    # 하이퍼파라미터 및 run 메타데이터 추적
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```

[run](./guides/runs)은 W&B의 기본 빌딩 블록입니다. 메트릭을 [추적하고](./guides/track), 로그를 [생성하며](./guides/artifacts), 작업을 [생성하는](./guides/launch) 등의 많은 작업에서 이를 자주 사용할 것입니다.

## 모든 것을 함께 결합하기

모든 것을 결합하면, 여러분의 트레이닝 스크립트는 다음 코드 예제와 유사할 수 있습니다. 강조된 코드는 W&B 전용 코드입니다. 기계학습 트레이닝을 모방하는 코드를 추가했음을 유의하세요.

```python
# train.py
import wandb
import random  # 데모 스크립트용

# 다음 줄 강조
wandb.login()

epochs = 10
lr = 0.01

# 강조 시작
run = wandb.init(
    # 이 run이 기록될 프로젝트 설정
    project="my-awesome-project",
    # 하이퍼파라미터 및 run 메타데이터 추적
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
)
# 강조 끝

offset = random.random() / 5
print(f"lr: {lr}")

# 트레이닝 run 시뮬레이션
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    # 다음 줄 강조
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()
```

이제 끝입니다! [https://wandb.ai/home](https://wandb.ai/home)에서 W&B App으로 이동하여 각 트레이닝 단계에서 W&B로 기록된 메트릭(정확도와 손실)이 어떻게 개선되었는지 확인하세요.

![위의 스크립트를 실행할 때마다 추적된 손실과 정확도를 보여줍니다. ](/images/quickstart/quickstart_image.png)

위의 이미지(클릭하여 확장)는 위의 스크립트를 실행할 때마다 추적된 손실과 정확도를 보여줍니다. 생성된 각 run 오브젝트는 **Runs** 열에 표시됩니다. 각 run 이름은 무작위로 생성됩니다.

## 다음은 무엇인가요?

W&B 에코시스템의 나머지를 탐색하세요.

1. [W&B 인테그레이션](guides/integrations)을 확인하여 PyTorch 같은 기계학습 프레임워크, Hugging Face 같은 ML 라이브러리, SageMaker 같은 ML 서비스와 W&B를 통합하는 방법을 배워보세요.
2. Run을 조직하고, 시각화를 임베드하고 자동화하며, 발견한 내용을 설명하고, [W&B Reports](./guides/reports)를 통해 협력자와 업데이트를 공유하세요.
3. [W&B Artifacts](./guides/artifacts) 생성하여 데이터셋, 모델, 종속성 및 결과를 기계학습 파이프라인의 각 단계에 따라 추적하세요.
4. [W&B Sweeps](./guides/sweeps)를 사용하여 하이퍼파라미터 검색을 자동화하고 가능한 모델 공간을 탐색하세요.
5. 데이터셋을 이해하고, 모델 예측을 시각화하며, 중앙 대시보드에서 인사이트를 공유하세요.
6. W&B AI 아카데미로 이동해서 LLM, MLOps 및 W&B Models를 실습 코스를 통해 배워보세요.

![](/images/quickstart/wandb_demo_experiments.gif)

## 자주 묻는 질문

**API 키를 어디서 찾을 수 있나요?**
www.wandb.ai에 로그인하면, API 키는 [인증 페이지](https://wandb.ai/authorize)에 있습니다.

**자동화된 환경에서 W&B를 어떻게 사용하나요?**
Google의 CloudML처럼 셸 명령어 실행이 불편한 자동화된 환경에서 모델을 트레이닝하는 경우에는 [환경 변수](guides/track/environment-variables) 설정 가이드를 살펴보세요.

**로컬 온프레미스 설치를 제공하나요?**
네, [W&B를 비공개로 호스팅](guides/hosting/)하여 로컬 머신이나 프라이빗 클라우드에서 운영할 수 있습니다. 이를 보는 빠른 튜토리얼 노트북은 [여기](http://wandb.me/intro)를 참조하세요.

**임시로 wandb 로깅을 꺼두려면 어떻게 하나요?**
테스트 코드를 실행하며 wandb 동기화를 비활성화하려면, 환경 변수 [`WANDB_MODE=offline`](./guides/track/environment-variables)으로 설정하세요.