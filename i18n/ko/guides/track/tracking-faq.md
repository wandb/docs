---
description: Answers to frequently asked question about W&B Experiments.
displayed_sidebar: default
---

# 실험 FAQ

<head>
  <title>실험에 관한 자주 묻는 질문</title>
</head>

다음 질문들은 W&B 아티팩트에 대해 자주 묻는 질문들입니다.

### 하나의 스크립트에서 여러 실행을 어떻게 시작하나요?

하나의 스크립트에서 여러 실행을 기록하기 위해 `wandb.init`과 `run.finish()`를 사용하세요:

1. `run = wandb.init(reinit=True)`: 실행을 재초기화할 수 있도록 이 설정을 사용하세요
2. `run.finish()`: 실행의 로깅을 마칠 때 이것을 사용하세요

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    for y in range(100):
        wandb.log({"metric": x + y})
    run.finish()
```

또는 자동으로 로깅을 마치는 파이썬 컨텍스트 관리자를 사용할 수 있습니다:

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    with run:
        for y in range(100):
            run.log({"metric": x + y})
```

### `InitStartError: wandb 프로세스와 통신하는 데 오류가 발생했습니다` <a href="#init-start-error" id="init-start-error"></a>

이 오류는 라이브러리가 서버로 데이터를 동기화하는 프로세스를 시작하는 데 어려움을 겪고 있음을 나타냅니다.

다음 해결책은 특정 환경에서 문제를 해결하는 데 도움이 될 수 있습니다:

<Tabs
  defaultValue="linux"
  values={[
    {label: '리눅스 및 OS X', value: 'linux'},
    {label: '구글 콜랩', value: 'google_colab'},
  ]}>
  <TabItem value="linux">

```python
wandb.init(settings=wandb.Settings(start_method="fork"))
```
</TabItem>
  <TabItem value="google_colab">

`0.13.0` 버전 이전에는 다음을 사용하는 것이 좋습니다:

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```
  </TabItem>
</Tabs>

### 분산 학습과 같은 멀티프로세싱에서 wandb를 어떻게 사용하나요?

학습 프로그램이 여러 프로세스를 사용하는 경우 `wandb.init()`을 실행하지 않은 프로세스에서 wandb 메서드 호출을 피하기 위해 프로그램을 구조화해야 합니다.\
\
멀티프로세스 학습을 관리하는 몇 가지 접근 방식이 있습니다:

1. 모든 프로세스에서 `wandb.init`을 호출하고, [group](../runs/grouping.md) 키워드 인수를 사용하여 공유 그룹을 정의합니다. 각 프로세스는 자체 wandb 실행을 가지며 UI는 학습 프로세스를 함께 그룹화합니다.
2. 하나의 프로세스에서만 `wandb.init`을 호출하고 [멀티프로세싱 큐](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes)를 통해 로깅할 데이터를 전달합니다.

:::info
Torch DDP를 사용한 코드 예제를 포함하여 이 두 가지 접근 방식에 대한 자세한 내용은 [분산 학습 가이드](./log/distributed-training.md)를 확인하세요.
:::

### 실행 이름을 프로그래밍 방식으로 어떻게 액세스하나요?

[`wandb.Run`](../../ref/python/run.md)의 `.name` 속성으로 사용 가능합니다.

```python
import wandb

wandb.init()
run_name = wandb.run.name
```

### 실행 이름을 실행 ID로 설정할 수 있나요?

실행 이름(예: snowy-owl-10)을 실행 ID(예: qvlp96vk)로 덮어쓰고 싶다면 이 코드 조각을 사용할 수 있습니다:

```python
import wandb

wandb.init()
wandb.run.name = wandb.run.id
wandb.run.save()
```

### 실행에 이름을 지정하지 않았습니다. 실행 이름은 어디서 오나요?

실행에 명시적으로 이름을 지정하지 않으면, UI에서 실행을 식별하는 데 도움이 되는 무작위 실행 이름이 실행에 할당됩니다. 예를 들어, 무작위 실행 이름은 "pleasant-flower-4"나 "misunderstood-glade-2"와 같이 보일 것입니다.

### 실행과 관련된 git 커밋을 어떻게 저장하나요?

스크립트에서 `wandb.init`을 호출하면, 원격 저장소에 대한 링크와 최신 커밋의 SHA를 포함한 git 정보를 자동으로 찾아 저장합니다. git 정보는 [실행 페이지](../app/pages/run-page.md)에 표시됩니다. 거기에 나타나지 않는 경우 스크립트를 실행할 때 셸의 현재 작업 디렉터리가 git으로 관리되는 폴더에 있는지 확인하세요.

실험을 실행하는 데 사용된 git 커밋과 명령은 사용자에게만 보이며, 외부 사용자에게는 숨겨져 있으므로 공개 프로젝트를 가지고 있다면 이러한 세부 정보는 비공개로 유지됩니다.

### 오프라인에서 메트릭을 저장하고 나중에 W&B에 동기화할 수 있나요?

기본적으로, `wandb.init`은 우리의 클라우드 호스팅 앱에 실시간으로 메트릭을 동기화하는 프로세스를 시작합니다. 기기가 오프라인 상태이거나 인터넷 액세스가 없거나 업로드를 보류하고 싶다면, 다음은 `wandb`를 오프라인 모드에서 실행하고 나중에 동기화하는 방법입니다.

두 개의 [환경 변수](./environment-variables.md)를 설정해야 합니다.

1. `WANDB_API_KEY=$KEY`, 여기서 `$KEY`는 [설정 페이지](https://app.wandb.ai/settings)에서의 API 키입니다.
2. `WANDB_MODE="offline"`

스크립트에서 이것이 어떻게 보일지에 대한 샘플입니다:

```python
import wandb
import os

os.environ["WANDB_API_KEY"] = YOUR_KEY_HERE
os.environ["WANDB_MODE"] = "offline"

config = {
    "dataset": "CIFAR10",
    "machine": "offline cluster",
    "model": "CNN",
    "learning_rate": 0.01,
    "batch_size": 128,
}

wandb.init(project="offline-demo")

for i in range(100):
    wandb.log({"accuracy": i})
```

터미널 출력 샘플:

![](/images/experiments/sample_terminal_output.png)

준비가 되면, 클라우드로 그 폴더를 보내는 동기화 명령을 실행하기만 하면 됩니다.

```shell
wandb sync wandb/dryrun-folder-name
```

![](/images/experiments/sample_terminal_output_cloud.png)

### wandb.init 모드의 차이점은 무엇인가요?

모드는 "online", "offline", 또는 "disabled"이며 기본값은 online입니다.

`online`(기본값): 이 모드에서 클라이언트는 데이터를 wandb 서버로 보냅니다.

`offline`: 이 모드에서 클라이언트는 wandb 서버로 데이터를 보내는 대신 데이터를 로컬 머신에 저장합니다. 이 데이터는 나중에 [`wandb sync`](../../ref/cli/wandb-sync.md) 명령으로 동기화될 수 있습니다.

`disabled`: 이 모드에서 클라이언트는 모의 객체를 반환하고 모든 네트워크 통신을 방지합니다. 클라이언트는 본질적으로 아무것도 하지 않는 것처럼 작동합니다. 즉, 모든 로깅이 완전히 비활성화됩니다. 그러나 모든 API 메서드의 스텁은 여전히 호출할 수 있습니다. 이 모드는 주로 테스트에서 사용됩니다.

### 내 실행의 상태는 UI에서 "충돌"로 표시되지만 내 기계에서 여전히 실행 중입니다. 내 데이터를 어떻게 되찾나요?

훈련 중에 기기와의 연결이 끊어졌을 가능성이 큽니다. [`wandb sync [PATH_TO_RUN]`](../../ref/cli/wandb-sync.md)을 실행하여 데이터를 복구할 수 있습니다. 실행 중인 실행의 실행 ID에 해당하는 `wandb` 디렉터리의 폴더가 실행 경로가 될 것입니다.

### `LaunchError: 권한이 거부되었습니다`

`Launch Error: 권한이 거부되었습니다` 오류 메시지가 나타나면, 실행하려는 프로젝트에 로그를 기록할 권한이 없습니다. 이는 몇 가지 다른 이유로 발생할 수 있습니다.

1. 이 기계에서 로그인하지 않았습니다. 명령 줄에서 [`wandb login`](../../ref/cli/wandb-login.md)을 실행하세요.
2. 존재하지 않는 엔터티를 설정했습니다. "엔터티"는 사용자 이름 또는 기존 팀의 이름이어야 합니다. 팀을 만들려면 [구독 페이지](https://app.wandb.ai/billing)로 이동하세요.
3. 프로젝트 권한이 없습니다. 프로젝트의 생성자에게 프로젝트의 개인 정보를 **Open**으로 설정하여 이 프로젝트에 실행을 기록할 수 있도록 요청하세요.

### W&B는 `multiprocessing` 라이브러리를 사용하나요?

네, W&B는 `multiprocessing` 라이브러리를 사용합니다. 다음과 같은 오류 메시지가 표시되면:

```
현재 프로세스의 부트스트래핑 단계가 끝나기 전에 새 프로세스를 시작하려고 시도했습니다.
```

이는 스크립트에서 W&B를 직접 실행하려고 할 때 `if __name__ == '__main__'`의 진입점 보호를 추가해야 할 수 있음을 의미할 수 있습니다.