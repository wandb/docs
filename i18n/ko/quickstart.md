---
description: W&B Quickstart.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 퀵스타트

W&B를 설치하고 몇 분 안에 머신러닝 실험 추적을 시작하세요.

## 1. 계정 생성 및 W&B 설치
시작하기 전에 계정을 생성하고 W&B를 설치해야 합니다:

1. [https://wandb.ai/site](https://wandb.ai/site)에서 무료 계정에 [가입](https://wandb.ai/site)한 후 wandb 계정에 로그인하세요.
2. [`pip`](https://pypi.org/project/wandb/)를 사용하여 Python 3 환경에서 컴퓨터에 wandb 라이브러리를 설치하세요.

다음 코드조각은 W&B CLI 및 Python 라이브러리를 사용하여 W&B에 설치하고 로그인하는 방법을 보여줍니다:

<Tabs
  defaultValue="notebook"
  values={[
    {label: '노트북', value: 'notebook'},
    {label: '커맨드라인', value: 'cli'},
  ]}>
  <TabItem value="cli">

Weights and Biases API와 상호작용하기 위한 CLI 및 Python 라이브러리 설치:

```bash
pip install wandb
```

  </TabItem>
  <TabItem value="notebook">

Weights and Biases API와 상호작용하기 위한 CLI 및 Python 라이브러리 설치:

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
    {label: '커맨드라인', value: 'cli'},
  ]}>
  <TabItem value="cli">

다음으로 W&B에 로그인하세요:

```bash
wandb login
```

또는 [W&B 서버](./guides/hosting)를 사용하는 경우(**Dedicated 클라우드** 또는 **Self-managed** 포함):

```bash
wandb login --relogin --host=http://your-shared-local-host.com
```

필요한 경우 배포 관리자에게 호스트명을 문의하세요.

메세지가 나타나면 [API 키](https://wandb.ai/authorize)를 입력하세요.

  </TabItem>
  <TabItem value="notebook">

다음으로 W&B Python SDK를 가져오고 로그인하세요:

```python
wandb.login()
```

메세지가 나타나면 [API 키](https://wandb.ai/authorize)를 입력하세요.
  </TabItem>
</Tabs>

## 3. run 시작 및 하이퍼파라미터 추적

Python 스크립트나 노트북에서 [`wandb.init()`](./ref/python/run.md)으로 W&B Run 오브젝트를 초기화하고, 하이퍼파라미터 이름과 값의 키-값 페어를 `config` 파라미터에 딕셔너리로 전달하세요:

```python
run = wandb.init(
    # 이 run이 기록될 프로젝트 설정
    project="my-awesome-project",
    # 하이퍼파라미터와 run 메타데이터 추적
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```

[run](./guides/runs)은 W&B의 기본 구성 요소입니다. [메트릭 추적](./guides/track), [로그 생성](./guides/artifacts), [job 생성](./guides/launch) 등을 자주 사용하게 될 것입니다.

## 모든 것을 통합하기

이 모든 것을 종합하면 훈련 스크립트는 다음 코드 예제와 비슷하게 보일 수 있습니다. 하이라이트된 코드는 W&B 관련 코드입니다.
머신러닝 훈련을 모방하는 코드가 추가되어있는 점 참고하세요.

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
    # 이 run이 기록될 프로젝트 설정
    project="my-awesome-project",
    # 하이퍼파라미터와 run 메타데이터 추적
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
)
# highlight-end

offset = random.random() / 5
print(f"lr: {lr}")

# 트레이닝 run을 시뮬레이션
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    # highlight-next-line
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()
```

다 되었습니다! [https://wandb.ai/home](https://wandb.ai/home)에서 W&B로 로그한 메트릭(정확도 및 손실)이 각 트레이닝 단계에서 어떻게 개선되었는지 확인하세요.

![위에 있는 스크립트를 실행할 때마다 추적된 손실과 정확도를 보여줍니다.](/images/quickstart/quickstart_image.png)

위의 이미지(확대하려면 클릭)는 위에 있는 스크립트를 실행할 때마다 추적된 손실과 정확도를 보여주고 있습니다. 생성된 각 run 오브젝트는 **Runs** 열에 표시됩니다. 각 run 이름은 무작위로 생성됩니다.

## 다음 단계는 무엇인가요?

W&B 에코시스템의 나머지 부분을 살펴보세요.

1. PyTorch와 같은 ML 프레임워크, Hugging Face와 같은 ML 라이브러리, 또는 SageMaker와 같은 ML 서비스와 W&B를 통합하는 방법을 알아보려면 [W&B Integrations](guides/integrations)를 확인하세요.
2. [W&B Reports](./guides/reports)로 run을 정리하고, 시각화를 내장하고 자동화하며, 발견한 내용을 설명하고, 협업자와 업데이트 사항을 공유하세요.
2. [W&B Artifacts](./guides/artifacts)를 생성하여 머신러닝 파이프라인의 각 단계에서의 데이터셋, 모델, 의존성 및 결과를 추적하세요.
3. [W&B Sweeps](./guides/sweeps)로 하이퍼파라미터 검색을 자동화하고 모델의 가능성을 탐색하세요.
4. 데이터셋을 이해하고, 모델 예측값을 시각화하고, [중앙 대시보드](./guides/tables)에서 인사이트을 공유하세요.


![](/images/quickstart/wandb_demo_experiments.gif)

## 자주 묻는 질문

**API 키는 어디에서 찾나요?**
API 키는 www.wandb.ai에 로그하면 [Authorize 페이지](https://wandb.ai/authorize)에 있습니다.

**자동화된 환경에서 W&B를 어떻게 사용하나요?**
Google의 CloudML과 같이 셸 명령어를 실행하기 불편한 자동화된 환경에서 모델을 트레이닝하는 경우, [환경 변수](guides/track/environment-variables)로 구성하는 방법에 대한 가이드를 참조하여 주시기 바랍니다.

**로컬, 온프레미스 설치를 제공하나요?**
네, 자체 머신이나 프라이빗 클라우드에 W&B를 로컬로 [프라이빗하게 호스팅](guides/hosting/)할 수 있으며, 그 방법은 [이 퀵 튜토리얼 노트북](http://wandb.me/intro)을 통해 확인해 보세요. 참고로, wandb 로컬 서버에 로그인하려면 로컬 인스턴스의 어드레스로 [호스트 플래그를 설정](guides/hosting/how-to-guides/basic-setup)할 수 있습니다.

**wandb 로깅을 임시로 끄려면 어떻게 하나요?**
코드를 테스트하는 중에 wandb 동기화를 비활성화하려면 환경 변수 [`WANDB_MODE=offline`](./guides/track/environment-variables)을 설정하세요.