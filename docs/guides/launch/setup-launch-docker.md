---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Docker 설정하기

다음 가이드는 로컬 기계에서 런치 에이전트 환경과 대상 리소스의 대기열에 대해 Docker를 사용하도록 W&B Launch를 구성하는 방법을 설명합니다.

동일한 로컬 기계에서 작업을 실행하고 런치 에이전트의 환경으로 Docker를 사용하는 것은 클러스터 관리 시스템(예: Kubernetes)이 없는 기계에 컴퓨팅이 설치된 경우 특히 유용합니다.

또한 Docker 대기열을 사용하여 강력한 작업장에서 작업을 실행할 수 있습니다.

:::tip
이 설정은 로컬 기계에서 실험을 수행하는 사용자나 원격 기계에 SSH로 접속하여 런치 작업을 제출하는 사용자에게 일반적입니다.
:::

W&B Launch와 함께 Docker를 사용할 때, W&B는 먼저 이미지를 빌드한 다음 해당 이미지에서 컨테이너를 빌드하고 실행합니다. 이미지는 Docker `docker run <image-uri>` 명령으로 빌드됩니다. 대기열 구성은 `docker run` 명령에 전달되는 추가 인수로 해석됩니다.

## Docker 대기열 구성하기

런치 대기열 구성(도커 대상 리소스용)은 [`docker run`](../../ref/cli/wandb-docker-run.md) CLI 명령에서 정의된 동일한 옵션을 수용합니다.

에이전트는 대기열 구성에서 정의된 옵션을 받습니다. 그런 다음 에이전트는 받은 옵션을 런치 작업의 구성에서 오버라이드와 병합하여 대상 리소스(이 경우 로컬 기계)에서 실행되는 최종 `docker run` 명령을 생성합니다.

두 가지 구문 변환 작업이 발생합니다:

1. 반복 옵션은 대기열 구성에서 리스트로 정의됩니다.
2. 플래그 옵션은 대기열 구성에서 값 `true`를 가진 부울로 정의됩니다.

예를 들어, 다음 대기열 구성:

```json
{
  "env": ["MY_ENV_VAR=value", "MY_EXISTING_ENV_VAR"],
  "volume": "/mnt/datasets:/mnt/datasets",
  "rm": true,
  "gpus": "all"
}
```

다음 `docker run` 명령으로 이어집니다:

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri> \
  --gpus all
```

볼륨은 문자열의 리스트 또는 단일 문자열로 지정할 수 있습니다. 여러 볼륨을 지정하는 경우 리스트를 사용하세요.

Docker는 자동으로 런치 에이전트 환경에서 값을 할당받지 않은 환경 변수를 전달합니다. 이는 런치 에이전트에 환경 변수 `MY_EXISTING_ENV_VAR`이 있는 경우, 해당 환경 변수가 컨테이너에서 사용 가능함을 의미합니다. 이는 대기열 구성에 키를 공개하지 않고 다른 구성 키를 사용하고자 할 때 유용합니다.

`docker run` 명령의 `--gpus` 플래그를 사용하면 Docker 컨테이너에서 사용할 수 있는 GPU를 지정할 수 있습니다. `gpus` 플래그 사용 방법에 대한 자세한 내용은 [Docker 문서](https://docs.docker.com/config/containers/resource_constraints/#gpu)를 참조하세요.

:::tip
* Docker 컨테이너 내에서 GPU를 사용하려면 [NVIDIA 컨테이너 툴킷](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)을 설치하세요.
* 코드 또는 아티팩트 기반 작업에서 이미지를 빌드하는 경우, [에이전트](#configure-a-launch-agent-on-a-local-machine)가 사용하는 기본 이미지를 NVIDIA 컨테이너 툴킷을 포함하도록 오버라이드할 수 있습니다.
  예를 들어, 런치 대기열에서 기본 이미지를 `tensorflow/tensorflow:latest-gpu`로 오버라이드할 수 있습니다:

  ```json
  {
    "builder": {
      "accelerator": {
        "base_image": "tensorflow/tensorflow:latest-gpu"
      }
    }
  }
  ```
:::

## 대기열 생성하기

W&B CLI를 사용하여 Docker를 컴퓨트 리소스로 사용하는 대기열을 생성하세요:

1. [Launch 페이지](https://wandb.ai/launch)로 이동하세요.
2. **Create Queue** 버튼을 클릭하세요.
3. 대기열을 생성할 **Entity**를 선택하세요.
4. **Name** 필드에 대기열의 이름을 입력하세요.
5. **Resource**로 **Docker**를 선택하세요.
6. **Configuration** 필드에 Docker 대기열 구성을 정의하세요.
7. **Create Queue** 버튼을 클릭하여 대기열을 생성하세요.

## 로컬 기계에서 런치 에이전트 구성하기

`launch-config.yaml`라는 이름의 YAML 구성 파일로 런치 에이전트를 구성하세요. 기본적으로 W&B는 `~/.config/wandb/launch-config.yaml`에서 구성 파일을 확인합니다. 런치 에이전트를 활성화할 때 다른 디렉터리를 선택적으로 지정할 수 있습니다.

:::tip
런치 에이전트의 핵심 구성 가능 옵션(최대 작업 수, W&B 엔터티, 런치 대기열 대신 구성 YAML 파일)을 W&B CLI를 사용하여 지정할 수 있습니다. 자세한 내용은 [`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) 명령을 참조하세요.
:::

## 핵심 에이전트 구성 옵션

다음 탭은 W&B CLI와 YAML 구성 파일로 핵심 구성 에이전트 옵션을 지정하는 방법을 보여줍니다:

<Tabs
defaultValue="CLI"
values={[
{label: 'W&B CLI', value: 'CLI'},
{label: '구성 파일', value: 'config'},
]}>
<TabItem value="CLI">

```bash
wandb launch-agent -q <queue-name> --max-jobs <n>
```

  </TabItem>
  <TabItem value="config">

```yaml title="launch-config.yaml"
max_jobs: <n concurrent jobs>
queues:
	- <queue-name>
```

  </TabItem>
</Tabs>

## Docker 이미지 빌더

귀하의 기계에서 런치 에이전트는 Docker 이미지를 빌드하도록 구성될 수 있습니다. 기본적으로 이러한 이미지는 기계의 로컬 이미지 저장소에 저장됩니다. 런치 에이전트가 Docker 이미지를 빌드할 수 있도록 하려면, 런치 에이전트 구성에서 `builder` 키를 `docker`로 설정하세요:

```yaml title="launch-config.yaml"
builder:
	type: docker
```

에이전트가 Docker 이미지를 빌드하지 않고 대신 레지스트리에서 미리 빌드된 이미지를 사용하길 원한다면, 런치 에이전트 구성에서 `builder` 키를 `noop`로 설정하세요:

```yaml title="launch-config.yaml"
builder:
  type: noop
```

## 컨테이너 레지스트리

런치는 Dockerhub, Google Container Registry, Azure Container Registry, Amazon ECR과 같은 외부 컨테이너 레지스트리를 사용합니다.
빌드한 환경과 다른 환경에서 작업을 실행하려면 에이전트가 컨테이너 레지스트리에서 이미지를 가져올 수 있도록 구성하세요.

런치 에이전트를 클라우드 레지스트리에 연결하는 방법에 대해 자세히 알아보려면 [고급 에이전트 설정](./setup-agent-advanced.md#agent-configuration) 페이지를 참조하세요.