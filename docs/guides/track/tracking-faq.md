---
title: Experiments FAQ
description: W&B Experiments에 대한 자주 묻는 질문의 답변.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B Artifacts에 관한 자주 묻는 질문들은 다음과 같습니다.

### 한 스크립트에서 여러 Runs를 어떻게 실행하나요?

`wandb.init`과 `run.finish()`를 사용하여 한 스크립트에서 여러 Runs를 로그할 수 있습니다:

1. `run = wandb.init(reinit=True)`: run을 재초기화할 수 있도록 이 설정을 사용하세요.
2. `run.finish()`: run의 로그를 종료하려면 이 메소드를 run의 끝에서 사용하세요.

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    for y in range(100):
        wandb.log({"metric": x + y})
    run.finish()
```

또는 자동으로 로그를 종료하는 파이썬 컨텍스트 매니저를 사용할 수 있습니다:

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    with run:
        for y in range(100):
            run.log({"metric": x + y})
```

### `InitStartError: Error communicating with wandb process` <a href="#init-start-error" id="init-start-error"></a>

이 오류는 라이브러리가 서버에 데이터를 동기화하는 프로세스를 시작하는 데 어려움을 겪고 있음을 나타냅니다.

몇 가지 환경에서 문제를 해결할 수 있는 다음 해결 방법이 있습니다:

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

`0.13.0` 버전 이전의 경우 다음 명령어를 사용하는 것이 좋습니다:

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```
  </TabItem>
</Tabs>

### 멀티프로세싱, 예를 들면 분산 트레이닝에서 wandb를 어떻게 사용할 수 있나요?

트레이닝 프로그램이 여러 프로세스를 사용하는 경우 `wandb.init()`을 실행하지 않은 프로세스에서 wandb 메소드 호출을 피하도록 프로그램을 구조화해야 합니다.

멀티프로세스 트레이닝을 관리하는 몇 가지 접근 방식이 있습니다:

1. 모든 프로세스에서 `wandb.init`을 호출하고, 공유 그룹을 정의하기 위해 [group](../runs/grouping.md) 키워드 인수를 사용합니다. 각 프로세스는 자체 wandb run을 가지며 UI는 트레이닝 프로세스를 함께 그룹화합니다.
2. 단일 프로세스에서만 `wandb.init`을 호출하고 로그할 데이터를 [multiprocessing queues](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes)를 통해 전달합니다.

:::info
[Torch DDP와 함께하는 코드 예제가 포함된 이 두 가지 접근 방법에 대한 자세한 내용을 보려면, [Distributed Training Guide](./log/distributed-training.md)를 확인하세요.
:::

### 사람에게 읽기 쉬운 run 이름을 프로그래밍 방식으로 어떻게 엑세스하나요?

이는 [`wandb.Run`](../../ref/python/run.md)의 `.name` 속성으로 사용할 수 있습니다.

```python
import wandb

wandb.init()
run_name = wandb.run.name
```

### run 이름을 run ID로 설정할 수 있나요?

run 이름을 run ID로 덮어쓰고 싶다면 다음 코드 스니펫을 사용할 수 있습니다:

```python
import wandb

wandb.init()
wandb.run.name = wandb.run.id
wandb.run.save()
```

### 내 run에 이름을 지정하지 않았습니다. run 이름은 어디서 오는 건가요?

run에 명시적으로 이름을 지정하지 않으면, UI에서 run을 식별하는 데 도움이 되는 임의의 run 이름이 할당됩니다. 예를 들어 "pleasant-flower-4" 또는 "misunderstood-glade-2"와 같은 임의의 run 이름이 될 수 있습니다.

### 내 run과 관련된 git 커밋을 어떻게 저장할 수 있나요?

스크립트에서 `wandb.init`을 호출하면, 최신 커밋의 SHA와 원격 저장소에 대한 링크 등과 같은 git 정보를 자동으로 저장하려 합니다. git 정보는 [run 페이지](../app/pages/run-page.md)에 나타나야 합니다. 만약 그것이 나타나지 않는다면, 스크립트를 실행할 때 셸의 현재 작업 디렉토리가 git로 관리되는 폴더에 있는지 확인하세요.

git 커밋과 실험을 실행한 커맨드는 사용자에게는 표시되지만 외부 사용자에게는 숨겨집니다. 따라서 공개 프로젝트가 있는 경우에도 이러한 세부정보는 비공개로 유지됩니다.

### 메트릭을 오프라인으로 저장하고 나중에 W&B에 동기화할 수 있나요?

기본적으로 `wandb.init`은 메트릭을 실시간으로 클라우드 호스트 앱에 동기화하는 프로세스를 시작합니다. 만약 기기가 오프라인이거나 인터넷 엑세스가 없거나 업로드를 잠시 보류하고 싶다면, `wandb`를 오프라인 모드로 실행하고 나중에 동기화하는 방법은 다음과 같습니다.

두 개의 [환경 변수](./environment-variables.md)를 설정해야 합니다.

1. `WANDB_API_KEY=$KEY`, 여기서 `$KEY`는 [설정 페이지](https://app.wandb.ai/settings)의 API 키입니다.
2. `WANDB_MODE="offline"`

그리고 다음은 스크립트에서의 예제입니다:

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

여기 터미널 출력의 예시가 있습니다:

![](/images/experiments/sample_terminal_output.png)

준비가 되면 해당 폴더를 클라우드로 보내기 위해 다음 명령어로 동기화를 실행하세요.

```shell
wandb sync wandb/dryrun-folder-name
```

![](/images/experiments/sample_terminal_output_cloud.png)

### wandb.init 모드 간의 차이는 무엇인가요?

모드는 "online", "offline", "disabled"이며 기본값은 online입니다.

`online`(기본값): 이 모드에서는 클라이언트가 wandb 서버로 데이터를 전송합니다.

`offline`: 이 모드에서는 데이터가 wandb 서버로 전송되지 않고, 대신 나중에 [`wandb sync`](../../ref/cli/wandb-sync.md) 명령어로 동기화할 수 있도록 로컬 머신에 데이터를 저장합니다.

`disabled`: 이 모드에서는 클라이언트를 모의 오브젝트를 반환하고 모든 네트워크 통신을 방지합니다. 클라이언트는 사실상 아무 작업도 하지 않는 것처럼 행동합니다. 즉, 모든 로그가 완전히 비활성화됩니다. 그러나 API 메소드에 대한 대체는 여전히 호출할 수 있습니다. 이는 주로 테스트에서 사용됩니다.

### 내 run의 상태가 UI에서 "crashed"라고 표시되는데 기계에서는 여전히 실행 중입니다. 데이터를 복구하려면 어떻게 해야 하나요?

트레이닝 중에 기계와의 연결이 끊어졌을 가능성이 있습니다. [`wandb sync [PATH_TO_RUN]`](../../ref/cli/wandb-sync.md)를 실행하여 데이터를 복구할 수 있습니다. run으로 가는 경로는 진행 중인 run에 대응하는 Run ID가 있는 `wandb` 디렉토리의 폴더입니다.

### `LaunchError: Permission denied`

`Launch Error: Permission denied`라는 오류 메시지를 받는 경우, 원하는 프로젝트로 run을 전송할 권한이 없습니다. 이는 여러 가지 이유로 발생할 수 있습니다.

1. 이 기계에 로그인되어 있지 않습니다. 커맨드라인에서 [`wandb login`](../../ref/cli/wandb-login.md)을 실행하세요.
2. 존재하지 않는 엔티티를 설정했습니다. "Entity"는 사용자 이름이나 기존 팀의 이름이어야 합니다. 팀을 생성해야 한다면 [Subscriptions page](https://app.wandb.ai/billing)로 이동하세요.
3. 프로젝트 권한이 없습니다. 프로젝트 생성자에게 프로젝트를 **Open**으로 설정하여 이 프로젝트로 run를 로그할 수 있게 요청하세요.

### W&B는 `multiprocessing` 라이브러리를 사용하나요?

네, W&B는 `multiprocessing` 라이브러리를 사용합니다. 다음과 같은 오류 메시지가 표시되는 경우가 있습니다:

```
An attempt has been made to start a new process before the current process 
has finished its bootstrapping phase.
```

이는 `if name == main`이라는 시작점 보호를 추가해야 할 수 있음을 의미할 수 있습니다. 스크립트에서 W&B를 직접 실행하려고 할 때만 이 시작점 보호를 추가해야 한다는 점을 기억하세요.