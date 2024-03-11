---
description: Answers to frequently asked question about W&B Experiments.
displayed_sidebar: default
---

# 실험 FAQ

<head>
  <title>실험에 관한 자주 묻는 질문</title>
</head>

다음은 W&B Artifacts에 대해 자주 묻는 질문입니다.

### 하나의 스크립트에서 여러 실행을 어떻게 시작하나요?

하나의 스크립트에서 여러 실행을 로그하기 위해 `wandb.init`과 `run.finish()`를 사용하세요:

1. `run = wandb.init(reinit=True)`: 실행을 재초기화하는 것을 허용하기 위해 이 설정을 사용하세요
2. `run.finish()`: 해당 실행에 대한 로깅을 마치기 위해 실행의 끝에서 이것을 사용하세요

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    for y in range(100):
        wandb.log({"metric": x + y})
    run.finish()
```

또는 파이썬 컨텍스트 매니저를 사용하여 자동으로 로깅을 마칠 수 있습니다:

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    with run:
        for y in range(100):
            run.log({"metric": x + y})
```

### `InitStartError: wandb 프로세스와의 통신 오류` <a href="#init-start-error" id="init-start-error"></a>

이 오류는 라이브러리가 서버로 데이터를 동기화하는 프로세스를 시작하는 데 어려움을 겪고 있음을 나타냅니다.

특정 환경에서 문제를 해결하는 데 도움이 될 수 있는 다음의 해결 방법이 있습니다:

<Tabs
  defaultValue="linux"
  values={[
    {label: 'Linux and OS X', value: 'linux'},
    {label: 'Google Colab', value: 'google_colab'},
  ]}>
  <TabItem value="linux">

```python
wandb.init(settings=wandb.Settings(start_method="fork"))
```
</TabItem>
  <TabItem value="google_colab">

`0.13.0` 이전 버전의 경우 다음을 사용하는 것이 좋습니다:

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```
  </TabItem>
</Tabs>

### wandb를 멀티프로세싱, 예를 들어 분산 트레이닝과 함께 사용할 수 있나요?

트레이닝 프로그램이 여러 프로세스를 사용하는 경우, `wandb.init()`을 실행하지 않은 프로세스에서 wandb 메소드 호출을 피하기 위해 프로그램을 구조화해야 합니다.\
\
멀티프로세스 트레이닝을 관리하는 몇 가지 접근 방법이 있습니다:

1. 모든 프로세스에서 `wandb.init`을 호출하고, [group](../runs/grouping.md) 키워드 인수를 사용하여 공유 그룹을 정의하세요. 각 프로세스는 자체 wandb 실행을 가지며 UI는 트레이닝 프로세스를 함께 그룹화합니다.
2. 하나의 프로세스에서만 `wandb.init`을 호출하고 로그할 데이터를 [멀티프로세싱 큐](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes)를 통해 전달하세요.

:::info
이 두 가지 접근 방법에 대한 자세한 내용과 Torch DDP를 사용한 코드 예제는 [분산 트레이닝 가이드](./log/distributed-training.md)를 확인하세요.
:::

### 실행 이름을 프로그래밍 방식으로 어떻게 액세스하나요?

[`wandb.Run`](../../ref/python/run.md)의 `.name` 속성으로 사용할 수 있습니다.

```python
import wandb

wandb.init()
run_name = wandb.run.name
```

### 실행 이름을 실행 ID로 설정할 수 있나요?

실행 이름(예: snowy-owl-10)을 실행 ID(예: qvlp96vk)로 덮어쓰고 싶다면 이 스니펫을 사용할 수 있습니다:

```python
import wandb

wandb.init()
wandb.run.name = wandb.run.id
wandb.run.save()
```

### 실행을 명명하지 않았습니다. 실행 이름은 어디서 오는 건가요?

실행을 명시적으로 명명하지 않으면 "pleasant-flower-4" 또는 "misunderstood-glade-2"와 같은 무작위 실행 이름이 실행에 할당되어 UI에서 실행을 식별하는 데 도움이 됩니다.

### 실행과 관련된 git 커밋을 어떻게 저장하나요?

스크립트에서 `wandb.init`이 호출될 때 자동으로 git 정보를 찾아 저장합니다. 여기에는 원격 리포지토리에 대한 링크와 최신 커밋의 SHA가 포함됩니다. git 정보가 [실행 페이지](../app/pages/run-page.md)에 나타나지 않는 경우 스크립트를 실행할 때 셸의 현재 작업 디렉터리가 git으로 관리되는 폴더에 위치해 있는지 확인하세요.

git 커밋과 실험을 실행하는 데 사용된 코맨드는 사용자에게 표시되지만 외부 사용자에게는 숨겨져 있으므로 공개 프로젝트를 가지고 있는 경우 이러한 세부 정보는 비공개로 유지됩니다.

### 메트릭을 오프라인으로 저장하고 나중에 W&B에 동기화할 수 있나요?

기본적으로 `wandb.init`은 실시간으로 클라우드 호스팅 앱에 메트릭을 동기화하는 프로세스를 시작합니다. 기계가 오프라인 상태이거나 인터넷 액세스가 없거나 업로드를 미루고 싶다면, 여기에서 `wandb`를 오프라인 모드로 실행하고 나중에 동기화하는 방법입니다.

두 [환경 변수](./environment-variables.md)를 설정해야 합니다.

1. `WANDB_API_KEY=$KEY`, 여기에서 `$KEY`는 [설정 페이지](https://app.wandb.ai/settings)에서의 API 키입니다
2. `WANDB_MODE="offline"`

스크립트에서 이것이 어떻게 보일지의 예:

```python
import wandb
import os

os.environ["WANDB_API_KEY"] = 여기에_키
os.environ["WANDB_MODE"] = "offline"

config = {
    "dataset": "CIFAR10",
    "machine": "오프라인 클러스터",
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

준비가 되면, 그 폴더를 클라우드로 보내기 위해 동기화 코맨드를 실행하기만 하면 됩니다.

```shell
wandb sync wandb/dryrun-folder-name
```

![](/images/experiments/sample_terminal_output_cloud.png)

### wandb.init 모드의 차이점은 무엇인가요?

모드는 "online", "offline" 또는 "disabled"일 수 있으며 기본적으로 online입니다.

`online`(기본값): 이 모드에서 클라이언트는 데이터를 wandb 서버로 전송합니다.

`offline`: 이 모드에서 클라이언트는 wandb 서버로 데이터를 보내는 대신 로컬 머신에 데이터를 저장합니다. 이 데이터는 나중에 [`wandb sync`](../../ref/cli/wandb-sync.md) 코맨드로 동기화할 수 있습니다.

`disabled`: 이 모드에서 클라이언트는 모의 객체를 반환하고 모든 네트워크 통신을 방지합니다. 클라이언트는 본질적으로 아무 작업도 수행하지 않습니다. 즉, 모든 로깅이 완전히 비활성화됩니다. 그러나 API 메소드의 모든 스텁은 여전히 호출할 수 있습니다. 이는 주로 테스트에서 사용됩니다.

### 실행 상태가 UI에서 "충돌"로 표시되지만 내 기계에서 계속 실행 중입니다. 데이터를 되찾기 위해 어떻게 하나요?

트레이닝하는 동안 기계와의 연결이 끊어졌을 가능성이 높습니다. [`wandb sync [실행_경로]`](../../ref/cli/wandb-sync.md)를 실행하여 데이터를 복구할 수 있습니다. 실행 경로는 진행 중인 실행의 실행 ID에 해당하는 `wandb` 디렉토리의 폴더입니다.

### `LaunchError: 권한 거부됨`

`Launch Error: 권한 거부됨` 오류 메시지를 받는 경우, 실행을 보내려는 프로젝트에 로그 기록 권한이 없습니다. 이는 몇 가지 다른 이유로 발생할 수 있습니다.

1. 이 기계에서 로그인하지 않았습니다. 커맨드라인에서 [`wandb login`](../../ref/cli/wandb-login.md)을 실행하세요.
2. 존재하지 않는 엔티티를 설정했습니다. "엔티티"는 사용자 이름이나 기존 팀의 이름이어야 합니다. 팀을 생성해야 하는 경우, [구독 페이지](https://app.wandb.ai/billing)로 이동하세요.
3. 프로젝트 권한이 없습니다. 프로젝트 생성자에게 프로젝트의 개인 정보 설정을 **Open**으로 설정하여 이 프로젝트에 실행을 로그할 수 있도록 요청하세요.

### W&B는 `multiprocessing` 라이브러리를 사용하나요?

예, W&B는 `multiprocessing` 라이브러리를 사용합니다. 다음과 같은 오류 메시지를 보게 되면:

```
현재 프로세스가 부트스트래핑 단계를 마치기 전에 새 프로세스를 시작하려는 시도가 있었습니다.
```

이는 W&B를 스크립트에서 직접 실행하려고 할 때 진입점 보호 `if __name__ == '__main__'`을 추가해야 할 수 있음을 의미합니다.