---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Docker 설정하기

다음 가이드에서는 W&B Launch를 사용하여 로컬 머신에서 런치 에이전트 환경과 큐의 대상 리소스 모두에 대해 Docker를 구성하는 방법을 설명합니다.

로컬 머신에서 동일한 런치 에이전트의 환경으로 작업을 실행하고 Docker를 사용하는 것은 컴퓨트가 클러스터 관리 시스템(예: Kubernetes)이 없는 머신에 설치된 경우 특히 유용합니다.

또한 Docker 큐를 사용하여 강력한 워크스테이션에서 작업을 실행할 수 있습니다.

:::tip
이 설정은 로컬 머신에서 실험을 수행하거나, 런치 작업을 제출하기 위해 SSH로 원격 머신에 접속하는 사용자에게 일반적입니다.
:::

W&B Launch에서 Docker를 사용할 때 W&B는 먼저 이미지를 빌드한 다음 해당 이미지에서 컨테이너를 빌드하고 실행합니다. 이미지는 Docker `docker run <image-uri>` 코맨드로 빌드됩니다. 큐 설정은 `docker run` 코맨드에 전달된 추가 인수로 해석됩니다.

## Docker 큐 구성하기

런치 큐 구성(Docker 대상 리소스용)은 [`docker run`](../../ref/cli/wandb-docker-run.md) CLI 코맨드에서 정의된 동일한 옵션을 허용합니다.

에이전트는 큐 구성에서 정의된 옵션을 받습니다. 그런 다음 에이전트는 받은 옵션을 런치 작업의 설정에서 오버라이드와 병합하여 대상 리소스(이 경우 로컬 머신)에서 실행되는 최종 `docker run` 코맨드를 생성합니다.

두 가지 구문 변환 작업이 이루어집니다:

1. 반복 옵션은 큐 구성에서 목록으로 정의됩니다.
2. 플래그 옵션은 큐 구성에서 `true` 값으로 Boolean으로 정의됩니다.

예를 들어, 다음과 같은 큐 구성:

```json
{
  "env": ["MY_ENV_VAR=value", "MY_EXISTING_ENV_VAR"],
  "volume": "/mnt/datasets:/mnt/datasets",
  "rm": true,
  "gpus": "all"
}
```

다음 `docker run` 코맨드로 이어집니다:

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri> \
  --gpus all
```

볼륨은 문자열 목록이나 단일 문자열로 지정할 수 있습니다. 여러 볼륨을 지정하는 경우 목록을 사용하십시오.

Docker는 자동으로 값이 할당되지 않은 환경 변수를 런치 에이전트 환경에서 전달합니다. 이는 런치 에이전트에 `MY_EXISTING_ENV_VAR`이라는 환경 변수가 있는 경우 해당 환경 변수가 컨테이너에서 사용 가능함을 의미합니다. 이는 큐 구성에 공개하지 않고 다른 설정 키를 사용하고자 할 때 유용합니다.

`docker run` 코맨드의 `--gpus` 플래그를 사용하면 Docker 컨테이너에서 사용할 수 있는 GPU를 지정할 수 있습니다. `gpus` 플래그 사용 방법에 대한 자세한 내용은 [Docker 문서](https://docs.docker.com/config/containers/resource_constraints/#gpu)를 참조하십시오.

:::tip
* Docker 컨테이너 내에서 GPU를 사용하려면 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)을 설치하십시오.
* 코드 또는 아티팩트 기반 작업에서 이미지를 빌드하는 경우, [에이전트](#configure-a-launch-agent-on-a-local-machine)가 사용하는 기본 이미지를 NVIDIA Container Toolkit을 포함하도록 오버라이드할 수 있습니다.
  예를 들어, 런치 큐에서 기본 이미지를 `tensorflow/tensorflow:latest-gpu`로 오버라이드할 수 있습니다:

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

## 큐 생성하기

W&B CLI를 사용하여 Docker를 컴퓨트 리소스로 사용하는 큐를 생성하십시오:

1. [Launch 페이지](https://wandb.ai/launch)로 이동합니다.
2. **Create Queue** 버튼을 클릭합니다.
3. 큐를 생성할 **Entity**를 선택합니다.
4. **Name** 필드에 큐의 이름을 입력합니다.
5. **Resource**로 **Docker**를 선택합니다.
6. **Configuration** 필드에 Docker 큐 구성을 정의합니다.
7. **Create Queue** 버튼을 클릭하여 큐를 생성합니다.

## 로컬 머신에서 런치 에이전트 구성하기

`launch-config.yaml`이라는 이름의 YAML 구성 파일로 런치 에이전트를 구성하십시오. 기본적으로 W&B는 `~/.config/wandb/launch-config.yaml`에서 구성 파일을 확인합니다. 런치 에이전트를 활성화할 때 다른 디렉토리를 선택적으로 지정할 수 있습니다.

:::tip
런치 에이전트의 핵심 구성 옵션(최대 작업 수, W&B 엔티티 및 런치 큐)을 지정하기 위해 구성 YAML 파일 대신 W&B CLI를 사용할 수 있습니다. 자세한 내용은 [`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) 코맨드를 참조하십시오.
:::

## 핵심 에이전트 구성 옵션

다음 탭은 W&B CLI 및 YAML 구성 파일로 핵심 구성 에이전트 옵션을 지정하는 방법을 보여줍니다:

<Tabs
defaultValue="CLI"
values={[
{label: 'W&B CLI', value: 'CLI'},
{label: 'Config file', value: 'config'},
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

머신에서 런치 에이전트는 Docker 이미지를 빌드하도록 구성할 수 있습니다. 기본적으로 이러한 이미지는 머신의 로컬 이미지 저장소에 저장됩니다. 런치 에이전트가 Docker 이미지를 빌드하도록 활성화하려면, 런치 에이전트 구성에서 `builder` 키를 `docker`로 설정하십시오:

```yaml title="launch-config.yaml"
builder:
	type: docker
```

에이전트가 Docker 이미지를 빌드하지 않고, 대신 레지스트리에서 미리 빌드된 이미지를 사용하길 원하는 경우, 런치 에이전트 구성의 `builder` 키를 `noop`로 설정하십시오:

```yaml title="launch-config.yaml"
builder:
  type: noop
```

## 컨테이너 레지스트리

Launch는 Dockerhub, Google Container Registry, Azure Container Registry 및 Amazon ECR과 같은 외부 컨테이너 레지스트리를 사용합니다.
빌드한 환경과 다른 환경에서 작업을 실행하려면, 에이전트가 컨테이너 레지스트리에서 pull할 수 있도록 구성하십시오.

런치 에이전트를 클라우드 레지스트리와 연결하는 방법에 대해 자세히 알아보려면, [고급 에이전트 설정](./setup-agent-advanced.md#agent-configuration) 페이지를 참조하십시오.