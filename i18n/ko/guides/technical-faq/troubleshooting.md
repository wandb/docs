---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 문제 해결

### wandb가 충돌하면 내 학습 실행을 중단시킬 수 있나요?

저희에게 있어 매우 중요한 것은 절대로 여러분의 학습 실행을 방해하지 않는 것입니다. wandb가 어떤 이유로 충돌하더라도 여러분의 학습이 계속 진행될 수 있도록 wandb를 별도의 프로세스에서 실행합니다. 인터넷이 끊기면 wandb는 [wandb.ai](https://wandb.ai)로 데이터를 전송하는 것을 계속 재시도할 것입니다.

### 로컬에서 잘 학습되고 있는데 W&B에서 실행이 충돌로 표시되는 이유는 무엇인가요?

이는 대부분 연결 문제일 가능성이 높습니다 — 서버의 인터넷 액세스가 끊기고 W&B로 데이터 동기화가 중단되면, 우리는 재시도 기간 후 실행을 충돌로 표시합니다.

### 로깅이 내 학습을 막나요?

"로깅 함수는 지연되나요? 결과를 여러분의 서버로 전송하고 나서 로컬 작업을 계속하기 위해 네트워크에 의존하고 싶지 않습니다."

`wandb.log`를 호출하면 로컬 파일에 한 줄을 작성합니다; 네트워크 호출을 막지 않습니다. 여러분이 `wandb.init`을 호출할 때, 우리는 동일한 기계에서 새 프로세스를 시작하여 파일 시스템 변경을 감지하고 여러분의 학습 프로세스와 비동기적으로 웹 서비스와 통신합니다.

### wandb가 내 터미널이나 주피터 노트북 출력에 작성하는 것을 어떻게 멈추나요?

환경 변수 [`WANDB_SILENT`](../track/environment-variables.md)를 `true`로 설정하세요.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Jupyter Notebook', value: 'notebook'},
    {label: 'Command Line', value: 'command-line'},
  ]}>
  <TabItem value="python">

```python
os.environ["WANDB_SILENT"] = "true"
```

  </TabItem>
  <TabItem value="notebook">

```python
%env WANDB_SILENT=true
```

  </TabItem>
  <TabItem value="command-line">

```python
WANDB_SILENT=true
```

  </TabItem>
</Tabs>

### wandb로 작업을 어떻게 종료하나요?

키보드에서 `Ctrl+D`를 눌러 wandb로 구성된 스크립트를 중지하세요.

### 네트워크 문제를 어떻게 처리하나요?

SSL 또는 네트워크 오류를 보고 있다면: `wandb: Network error (ConnectionError), 재시도 루프에 들어갑니다`. 이 문제를 해결하기 위해 몇 가지 다른 접근 방법을 시도할 수 있습니다:

1. SSL 인증서를 업그레이드하세요. Ubuntu 서버에서 스크립트를 실행하는 경우, `update-ca-certificates`를 실행하세요. 유효한 SSL 인증서가 없으면 보안 취약점 때문에 학습 로그를 동기화할 수 없습니다.
2. 네트워크가 불안정한 경우, [오프라인 모드](../track/launch.md)에서 학습을 실행하고 인터넷 액세스가 있는 기계에서 파일을 우리에게 동기화하세요.
3. [W&B 프라이빗 호스팅](../hosting/intro.md)을 실행해 보세요. 이는 여러분의 기계에서 운영되며 파일을 우리의 클라우드 서버에 동기화하지 않습니다.

`SSL CERTIFICATE_VERIFY_FAILED`: 이 오류는 회사의 방화벽 때문일 수 있습니다. 로컬 CA를 설정한 다음 사용할 수 있습니다:

`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`

### 모델을 학습하는 동안 인터넷 연결이 끊기면 어떻게 되나요?

라이브러리가 인터넷에 연결할 수 없는 경우, 재시도 루프에 들어가 메트릭 스트리밍을 네트워크가 복구될 때까지 계속 시도할 것입니다. 이 기간 동안 여러분의 프로그램은 계속 실행될 수 있습니다.

인터넷이 없는 기계에서 실행해야 하는 경우, `WANDB_MODE=offline`을 설정하여 메트릭이 여러분의 하드 드라이브에 로컬로만 저장되도록 할 수 있습니다. 나중에 `wandb sync DIRECTORY`를 호출하여 데이터를 우리 서버로 스트리밍할 수 있습니다.