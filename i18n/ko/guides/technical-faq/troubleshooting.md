---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 문제 해결

### wandb가 충돌하면, 내 트레이닝 실행을 충돌시킬 수 있나요?

트레이닝 실행에 절대 방해가 되지 않도록 하는 것이 우리에게 매우 중요합니다. wandb가 어떤 이유로 충돌하더라도 귀하의 트레이닝은 계속 실행될 수 있도록 wandb를 별도의 프로세스에서 실행합니다. 인터넷이 끊어지면 wandb는 [wandb.ai](https://wandb.ai)로 데이터를 전송하기 위해 계속해서 재시도합니다.

### 로컬에서 잘 트레이닝되는데 W&B에서 실행이 충돌로 표시되는 이유는 무엇인가요?

이는 대개 연결 문제입니다 — 서버가 인터넷 접속을 잃고 W&B로 데이터 동기화가 중단되면, 우리는 재시도 후 짧은 기간이 지나면 실행을 충돌로 표시합니다.

### 로깅이 내 트레이닝을 차단하나요?

"로깅 함수가 느리게 작동하나요? 결과를 귀하의 서버로 전송한 후에 로컬 작업을 계속하려면 네트워크에 의존하고 싶지 않습니다."

`wandb.log`를 호출하면 로컬 파일에 줄을 작성합니다; 네트워크 호출을 차단하지 않습니다. `wandb.init`를 호출할 때, 우리는 같은 기계에서 새 프로세스를 시작하여 파일시스템 변경을 감지하고 귀하의 트레이닝 프로세스와는 비동기적으로 웹 서비스와 통신합니다.

### wandb가 내 터미널이나 주피터 노트북 출력에 작성하는 것을 어떻게 멈출까요?

환경 변수 [`WANDB_SILENT`](../track/environment-variables.md)를 `true`로 설정하세요.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Jupyter Notebook', value: 'notebook'},
    {label: '커맨드라인', value: 'command-line'},
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

### wandb를 사용하는 작업을 어떻게 종료하나요?

키보드에서 `Ctrl+D`를 눌러 wandb로 구성된 스크립트를 중지하세요.

### 네트워크 문제를 어떻게 다뤄야 하나요?

SSL 또는 네트워크 오류를 보고 있다면:`wandb: 네트워크 오류 (ConnectionError), 재시도 루프 진입`. 이 문제를 해결하기 위해 몇 가지 다른 접근 방법을 시도해볼 수 있습니다:

1. SSL 인증서를 업그레이드하세요. Ubuntu 서버에서 스크립트를 실행하는 경우, `update-ca-certificates`를 실행하세요. 유효한 SSL 인증서 없이는 보안 취약점 때문에 트레이닝 로그를 동기화할 수 없습니다.
2. 네트워크가 불안정하면, 트레이닝을 [오프라인 모드](../track/launch.md)에서 실행하고 인터넷 접속이 가능한 기계에서 파일을 우리에게 동기화하세요.
3. [W&B 프라이빗 호스팅](../hosting/intro.md)을 실행해보세요. 이는 귀하의 기계에서 작동하며 파일을 우리의 클라우드 서버로 동기화하지 않습니다.

`SSL CERTIFICATE_VERIFY_FAILED`: 이 오류는 귀하의 회사 방화벽 때문일 수 있습니다. 로컬 CA를 설정한 후 다음을 사용할 수 있습니다:

`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`

### 모델을 트레이닝하는 동안 인터넷 연결이 끊어지면 어떻게 되나요?

라이브러리가 인터넷에 연결할 수 없다면 재시도 루프에 들어가 네트워크가 복원될 때까지 메트릭 스트리밍을 계속 시도합니다. 이 기간 동안 귀하의 프로그램은 계속 실행될 수 있습니다.

인터넷이 없는 기계에서 실행해야 한다면 `WANDB_MODE=offline`을 설정하여 메트릭이 귀하의 하드 드라이브에 로컬로만 저장되도록 할 수 있습니다. 나중에 `wandb sync 디렉토리`를 호출하여 데이터를 우리 서버로 스트리밍할 수 있습니다.