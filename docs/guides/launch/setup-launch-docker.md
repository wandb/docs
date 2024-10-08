---
title: Tutorial: Set up W&B Launch with Docker
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

다음 가이드는 W&B Launch를 구성하여 Docker를 로컬 머신에서 Launch 에이전트 환경과 큐의 대상 자원 모두에 사용할 수 있도록 하는 방법을 설명합니다.

Docker를 사용하여 작업을 실행하고 동일한 로컬 머신에서 Launch 에이전트의 환경으로 사용하는 것은 클러스터 관리 시스템(예: Kubernetes)이 없는 머신에 컴퓨팅이 설치된 경우 특히 유용합니다.

또한 Docker 큐를 사용하여 강력한 워크스테이션에서 작업 부하를 실행할 수 있습니다.

:::tip
이 설정은 로컬 머신에서 실험을 수행하거나, 원격 머신에 SSH하여 Launch 작업을 제출하는 사용자에게 일반적입니다.
:::

Docker를 W&B Launch와 함께 사용할 때, W&B는 먼저 이미지를 구축한 후 해당 이미지에서 컨테이너를 빌드하고 실행합니다. 이미지는 Docker `docker run <image-uri>` 명령어로 빌드됩니다. 큐 설정은 `docker run` 명령어에 전달되는 추가 인수로 해석됩니다.

## Docker 큐 구성하기

Docker 대상 자원을 위한 Launch 큐 설정은 [`docker run`](../../ref/cli/wandb-docker-run.md) CLI 명령어에 정의된 옵션과 동일한 옵션을 수용합니다.

에이전트는 큐 구성에 정의된 옵션을 수신합니다. 그런 다음, 에이전트는 수신한 옵션을 Launch 작업 설정의 임의 덮어쓰기와 병합하여 대상 자원(이 경우, 로컬 머신)에서 실행될 최종 `docker run` 명령을 생성합니다.

두 가지 문법 변환이 발생합니다:

1. 반복된 옵션은 큐 구성에서 목록으로 정의됩니다.
2. 플래그 옵션은 큐 구성에서 Boolean으로 값이 `true`로 정의됩니다.

예를 들어, 다음의 큐 구성:

```json
{
  "env": ["MY_ENV_VAR=value", "MY_EXISTING_ENV_VAR"],
  "volume": "/mnt/datasets:/mnt/datasets",
  "rm": true,
  "gpus": "all"
}
```

다음 `docker run` 명령어를 결과로 생성합니다:

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri> \
  --gpus all
```

볼륨은 문자열 목록으로 지정하거나, 단일 문자열로 지정할 수 있습니다. 여러 볼륨을 지정하는 경우 목록을 사용하세요.

Docker는 Launch 에이전트 환경에서 값이 할당되지 않은 환경 변수를 자동으로 전달합니다. 즉, Launch 에이전트가 환경 변수 `MY_EXISTING_ENV_VAR`를 가지고 있다면 해당 환경 변수는 컨테이너에서 사용할 수 있습니다. 이는 큐 구성에 게시하지 않고 다른 설정 키를 사용하려는 경우 유용합니다.

`docker run` 명령어의 `--gpus` 플래그는 Docker 컨테이너에서 사용할 수 있는 GPU를 지정할 수 있게 합니다. `gpus` 플래그 사용 방법에 대한 자세한 정보는 [Docker documentation](https://docs.docker.com/config/containers/resource_constraints/#gpu)를 참조하세요.

:::tip
* Docker 컨테이너 내에서 GPU를 사용하려면 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)을 설치하세요.
* 코드 또는 아티팩트 기반 작업에서 이미지를 빌드하는 경우, [에이전트](#configure-a-launch-agent-on-a-local-machine)를 NVIDIA Container Toolkit을 포함하도록 기본 이미지를 덮어쓸 수 있습니다.
  예를 들어, Launch 큐 내에서 기본 이미지를 `tensorflow/tensorflow:latest-gpu`로 덮어쓸 수 있습니다:

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

W&B CLI로 Docker를 컴퓨팅 자원으로 사용하는 큐를 생성하세요:

1. [Launch 페이지](https://wandb.ai/launch)로 이동하세요.
2. **Create Queue** 버튼을 클릭하세요.
3. 큐를 생성할 **Entity**를 선택하세요.
4. **Name** 필드에 큐의 이름을 입력하세요.
5. **Resource**로 **Docker**를 선택하세요.
6. **Configuration** 필드에 Docker 큐 설정을 정의하세요.
7. **Create Queue** 버튼을 클릭하여 큐를 생성하세요.

## 로컬 머신에서 Launch 에이전트 구성하기

`launch-config.yaml`이라는 YAML 설정 파일로 Launch 에이전트를 구성하세요. 기본적으로, W&B는 `~/.config/wandb/launch-config.yaml`에서 설정 파일을 확인합니다. Launch 에이전트를 활성화할 때 다른 디렉토리를 지정할 수 있습니다.

:::tip
W&B CLI를 사용하여 Launch 에이전트에 대한 핵심 구성 가능한 옵션을 지정할 수 있습니다(설정 YAML 파일 대신): 최대 작업 수, W&B 엔티티 및 Launch 큐. [`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) 명령을 참고하세요.
:::

## 핵심 에이전트 설정 옵션

다음 탭은 W&B CLI와 YAML 설정 파일로 핵심 설정 에이전트 옵션을 지정하는 방법을 보여줍니다:

<Tabs
defaultValue="CLI"
values={[
{label: 'W&B CLI', value: 'CLI'},
{label: 'Config file', value: 'config'}
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

머신의 Launch 에이전트는 Docker 이미지를 빌드하도록 구성할 수 있습니다. 기본적으로, 이러한 이미지는 머신의 로컬 이미지 저장소에 저장됩니다. Launch 에이전트에서 Docker 이미지를 빌드할 수 있도록 하려면, Launch 에이전트 설정의 `builder` 키를 `docker`로 설정하세요:

```yaml title="launch-config.yaml"
builder:
  type: docker
```

에이전트가 Docker 이미지를 빌드하도록 하지 않고, 대신 레지스트리에서 사전 빌드된 이미지를 사용하려면, Launch 에이전트 설정의 `builder` 키를 `noop`으로 설정하세요:

```yaml title="launch-config.yaml"
builder:
  type: noop
```

## 컨테이너 레지스트리

Launch는 Dockerhub, Google Container Registry, Azure Container Registry, Amazon ECR과 같은 외부 컨테이너 레지스트리를 사용합니다.  
빌드한 환경과 다른 환경에서 작업을 실행하려면, 에이전트를 구성하여 컨테이너 레지스트리에서 가져올 수 있도록 해야 합니다.

Launch 에이전트를 클라우드 레지스트리와 연결하는 방법에 대해 자세히 알아보려면 [Advanced agent setup](./setup-agent-advanced.md#agent-configuration) 페이지를 참조하세요.