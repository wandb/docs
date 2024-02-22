---
description: W&B Quickstart.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 퀵스타트

W&B를 설치하고 몇 분 안에 머신 러닝 실험을 추적하세요.

## 1. 계정 생성 및 W&B 설치
시작하기 전에 계정을 생성하고 W&B를 설치하세요:

1. [https://wandb.ai/site](https://wandb.ai/site)에서 무료 계정을 [가입](https://wandb.ai/site)한 다음 wandb 계정에 로그인하세요.  
2. [`pip`](https://pypi.org/project/wandb/)을 사용하여 Python 3 환경에서 컴퓨터에 wandb 라이브러리를 설치하세요.  


다음 코드 조각은 W&B CLI 및 Python 라이브러리를 사용하여 W&B에 설치하고 로그인하는 방법을 보여줍니다:

<Tabs
  defaultValue="notebook"
  values={[
    {label: '노트북', value: 'notebook'},
    {label: '명령 줄', value: 'cli'},
  ]}>
  <TabItem value="cli">

Weights and Biases API와 상호 작용하기 위한 CLI 및 Python 라이브러리를 설치하세요:

```bash
pip install wandb
```

  </TabItem>
  <TabItem value="notebook">

Weights and Biases API와 상호 작용하기 위한 CLI 및 Python 라이브러리를 설치하세요:

```notebook
!pip install wandb
```


  </TabItem>
</Tabs>

## 2. W&B에 로그인하기


<Tabs
  defaultValue="notebook"
  values={[
    {label: '노트북', value: 'notebook'},
    {label: '명령 줄', value: 'cli'},
  ]}>
  <TabItem value="cli">

다음으로, W&B에 로그인하세요:

```bash
wandb login
```

또는 [W&B 서버](./guides/hosting)를 사용하는 경우(**데디케이티드 클라우드** 또는 **자가 관리** 포함):

```bash
wandb login --relogin --host=http://your-shared-local-host.com
```

필요한 경우 배포 관리자에게 호스트 이름을 문의하세요.

요청 시 [API 키](https://wandb.ai/authorize)를 제공하세요.

  </TabItem>
  <TabItem value="notebook">

다음으로, W&B Python SDK를 가져와서 로그인하세요:

```python
wandb.login()
```

요청 시 [API 키](https://wandb.ai/authorize)를 제공하세요.
  </TabItem>
</Tabs>

## 3. 실행 시작 및 하이퍼파라미터 추적

Python 스크립트나 노트북에서 [`wandb.init()`](./ref/python/run.md)으로 W&B Run 객체를 초기화하고 `config` 파라미터에 하이퍼파라미터 이름과 값의 키-값 쌍이 포함된 사전을 전달하세요:

```python
run = wandb.init(
    # 이 실행이 기록될 프로젝트 설정
    project="my-awesome-project",
    # 하이퍼파라미터 및 실행 메타데이터 추적
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```


[실행](./guides/runs)은 W&B의 기본 구성 요소입니다. [메트릭 추적](./guides/track), [로그 생성](./guides/artifacts), [작업 생성](./guides/launch) 등을 자주 사용할 것입니다.

## 모든 것을 함께하기

모든 것을 함께하면, 학습 스크립트는 다음 코드 예와 유사하게 보일 수 있습니다. 하이라이트된 코드는 W&B 관련 코드를 보여줍니다.
머신 러닝 학습을 모방한 코드가 추가되었다는 점에 주의하세요.

```python
# train.py
import wandb
import random  # 데모 스크립트용

# highlight-next-line
wandb.login()

epochs = 10
lr = 0.01

# highlight-start
run = wandb.init(
    # 이 실행이 기록될 프로젝트 설정
    project="my-awesome-project",
    # 하이퍼파라미터 및 실행 메타데이터 추적
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
)
# highlight-end

offset = random.random() / 5
print(f"lr: {lr}")

# 학습 실행을 시뮬레이션
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    # highlight-next-line
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()
```

다 했습니다! [https://wandb.ai/home](https://wandb.ai/home)에서 W&B 앱으로 이동하여 W&B로 로그한 메트릭(정확도 및 손실)이 각 학습 단계에서 어떻게 개선되었는지 확인하세요.

![위에서 실행한 스크립트마다 추적한 손실과 정확도를 보여줍니다.](/images/quickstart/quickstart_image.png)

위 이미지(확대하려면 클릭)는 위에서 실행한 스크립트마다 추적한 손실과 정확도를 보여줍니다. 생성된 각 실행 객체는 **실행** 열에 나와 있으며, 각 실행 이름은 무작위로 생성됩니다.

## 다음 단계는?

W&B 생태계의 나머지 부분을 탐색하세요.

1. PyTorch와 같은 ML 프레임워크, Hugging Face와 같은 ML 라이브러리 또는 SageMaker와 같은 ML 서비스와 W&B를 통합하는 방법을 알아보려면 [W&B 통합](guides/integrations)을 확인하세요.
2. 실행을 구성하고, 시각화를 내장 및 자동화하고, 발견한 내용을 설명하고, 공동 작업자와 업데이트를 공유하는 데 [W&B 리포트](./guides/reports)를 사용하세요.
2. 머신 러닝 파이프라인의 각 단계를 통해 데이터세트, 모델, 의존성 및 결과를 추적하는 [W&B 아티팩트](./guides/artifacts)를 생성하세요.
3. 하이퍼파라미터 검색을 자동화하고 가능한 모델의 공간을 탐색하는 데 [W&B 스윕](./guides/sweeps)을 사용하세요.
4. 데이터세트를 이해하고, 모델 예측을 시각화하고, [중앙 대시보드](./guides/tables)에서 통찰력을 공유하세요.


![](/images/quickstart/wandb_demo_experiments.gif)

## 자주 묻는 질문

**API 키를 어디서 찾나요?**
www.wandb.ai에 로그인하면, API 키는 [인증 페이지](https://wandb.ai/authorize)에 있습니다.

**자동화된 환경에서 W&B를 어떻게 사용하나요?**
Google의 CloudML과 같이 셸 명령을 실행하기 불편한 자동화된 환경에서 모델을 학습하는 경우, [환경 변수](guides/track/environment-variables)를 사용한 구성 가이드를 확인해보세요.

**온-프레미스 설치를 제공하나요?**
예, 자체 기계나 프라이빗 클라우드에 W&B를 사설로 호스팅할 수 있으며, 방법을 보려면 [이 퀵 튜토리얼 노트북](http://wandb.me/intro)을 시도하세요. 주의하세요, wandb 로컬 서버에 로그인하려면 로컬 인스턴스의 주소로 [호스트 플래그를 설정](guides/hosting/how-to-guides/basic-setup)할 수 있습니다.

**Wandb 로깅을 일시적으로 끄려면 어떻게 하나요?**
코드를 테스트하고 wandb 동기화를 비활성화하려면 환경 변수 [`WANDB_MODE=offline`](./guides/track/environment-variables)을 설정하세요.