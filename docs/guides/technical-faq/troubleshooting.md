---
title: Troubleshooting
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

### wandb가 충돌하면 트레이닝 run도 충돌할 가능성이 있나요?

우리에게는 여러분의 트레이닝 runs에 결코 방해하지 않는 것이 매우 중요합니다. wandb가 어떻게든 충돌하더라도 여러분의 트레이닝은 계속될 수 있도록 wandb를 별도의 프로세스에서 실행합니다. 인터넷이 끊어져도, wandb는 [wandb.ai](https://wandb.ai)로 데이터를 보내려고 계속 시도할 것입니다.

### 로컬에서는 잘 트레이닝되고 있을 때, W&B에서 run이 충돌된 것으로 표시되는 이유는 무엇인가요?

이것은 연결 문제일 가능성이 큽니다. 만약 여러분의 서버가 인터넷 연결을 잃고 데이터를 W&B에 동기화하지 못하면, wandb는 재시도 후 짧은 기간이 지나면 run을 충돌로 표시합니다.

### 로그가 제 트레이닝을 차단하나요?

"로그 함수가 게으른가요? 네트워크에 의존해서 결과를 서버에 보내고 나서 제 로컬 작업을 계속하고 싶지 않습니다."

`wandb.log`를 호출하면 로컬 파일에 한 줄을 씁니다. 네트워크 호출을 차단하지 않습니다. `wandb.init`을 호출하면, 파일 시스템 변화를 감지하고 트레이닝 프로세스와 비동기로 웹 서비스와 통신하는 새로운 프로세스를 동일한 머신에서 시작합니다.

### wandb가 내 터미널 또는 jupyter 노트북 출력에 쓰지 않도록 하려면 어떻게 하나요?

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

### wandb로 작업을 중지하려면 어떻게 하나요?

wandb로 도구화된 스크립트를 중지하려면 키보드에서 `Ctrl+D`를 누르세요.

### 네트워크 문제를 어떻게 처리하나요?

SSL 또는 네트워크 오류가 보이면: `wandb: Network error (ConnectionError), entering retry loop`. 이 문제를 해결하기 위해 몇 가지 방법을 시도할 수 있습니다:

1. SSL 인증서를 업그레이드하세요. Ubuntu 서버에서 스크립트를 실행 중이라면, `update-ca-certificates`를 실행하세요. 유효한 SSL 인증서 없이는 트레이닝 로그를 동기화할 수 없으므로 이는 보안 취약점입니다.
2. 네트워크가 불안정하다면, [오프라인 모드](../track/launch.md)에서 트레이닝을 실행하고 인터넷이 연결된 머신에서 우리에게 파일을 동기화하세요.
3. [W&B Private Hosting](../hosting/intro.md)을 실행해 보세요. 이는 당신의 머신에서 작동하며 우리 클라우드 서버에 파일을 동기화하지 않습니다.

`SSL CERTIFICATE_VERIFY_FAILED`: 이 오류는 당신의 회사 방화벽 때문일 수 있습니다. 로컬 CA를 설정한 다음 다음을 사용할 수 있습니다:

`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`

### 트레이닝 중 인터넷 연결이 끊어지면 어떻게 되나요?

우리 라이브러리가 인터넷에 연결할 수 없으면 재시도 루프에 들어가고 네트워크가 복원될 때까지 메트릭을 스트리밍하려고 시도할 것입니다. 이 기간 동안에도 프로그램은 계속 실행될 수 있습니다.

인터넷 없이 머신에서 실행해야 한다면, `WANDB_MODE=offline`으로 설정하여 메트릭을 로컬 하드 드라이브에만 저장하게 할 수 있습니다. 나중에 `wandb sync DIRECTORY`를 호출하여 데이터를 서버로 전송할 수 있습니다.