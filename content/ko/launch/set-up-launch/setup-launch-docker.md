---
title: 'Tutorial: Set up W&B Launch with Docker'
menu:
  launch:
    identifier: ko-launch-set-up-launch-setup-launch-docker
    parent: set-up-launch
url: guides/launch/setup-launch-docker
---

다음 가이드는 로컬 장치에서 Docker를 사용하도록 W&B Launch를 구성하여 Launch 에이전트 환경과 대기열의 대상 리소스 모두에 대해 설명합니다.

Docker를 사용하여 작업을 실행하고 동일한 로컬 장치에서 Launch 에이전트의 환경으로 사용하는 것은 컴퓨팅이 Kubernetes와 같은 클러스터 관리 시스템이 없는 장치에 설치된 경우에 특히 유용합니다.

Docker 대기열을 사용하여 강력한 워크스테이션에서 워크로드를 실행할 수도 있습니다.

{{% alert %}}
이 설정은 로컬 장치에서 Experiments를 수행하거나 Launch 작업을 제출하기 위해 SSH로 연결하는 원격 장치를 사용하는 사용자에게 일반적입니다.
{{% /alert %}}

W&B Launch와 함께 Docker를 사용하면 W&B는 먼저 이미지를 빌드한 다음 해당 이미지에서 컨테이너를 빌드하고 실행합니다. 이미지는 Docker `docker run <image-uri>` 코맨드로 빌드됩니다. 대기열 설정은 `docker run` 코맨드에 전달되는 추가 인수로 해석됩니다.

## Docker 대기열 구성

Launch 대기열 설정(Docker 대상 리소스의 경우)은 [`docker run`]({{< relref path="/ref/cli/wandb-docker-run.md" lang="ko" >}}) CLI 코맨드에 정의된 것과 동일한 옵션을 허용합니다.

에이전트는 대기열 설정에 정의된 옵션을 수신합니다. 그런 다음 에이전트는 수신된 옵션을 Launch 작업의 설정에서 가져온 재정의와 병합하여 대상 리소스(이 경우 로컬 장치)에서 실행되는 최종 `docker run` 코맨드를 생성합니다.

다음과 같은 두 가지 구문 변환이 수행됩니다.

1. 반복되는 옵션은 대기열 설정에 목록으로 정의됩니다.
2. 플래그 옵션은 대기열 설정에 값 `true`가 있는 부울로 정의됩니다.

예를 들어, 다음 대기열 설정은 다음과 같습니다.

```json
{
  "env": ["MY_ENV_VAR=value", "MY_EXISTING_ENV_VAR"],
  "volume": "/mnt/datasets:/mnt/datasets",
  "rm": true,
  "gpus": "all"
}
```

다음과 같은 `docker run` 코맨드가 생성됩니다.

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri> \
  --gpus all
```

볼륨은 문자열 목록 또는 단일 문자열로 지정할 수 있습니다. 여러 볼륨을 지정하는 경우 목록을 사용하세요.

Docker는 값이 할당되지 않은 환경 변수를 Launch 에이전트 환경에서 자동으로 전달합니다. 즉, Launch 에이전트에 환경 변수 `MY_EXISTING_ENV_VAR`가 있는 경우 해당 환경 변수를 컨테이너에서 사용할 수 있습니다. 이는 대기열 설정에서 게시하지 않고 다른 구성 키를 사용하려는 경우에 유용합니다.

`docker run` 코맨드의 `--gpus` 플래그를 사용하면 Docker 컨테이너에서 사용할 수 있는 GPU를 지정할 수 있습니다. `gpus` 플래그를 사용하는 방법에 대한 자세한 내용은 [Docker 설명서](https://docs.docker.com/config/containers/resource_constraints/#gpu)를 참조하세요.

{{% alert %}}
* Docker 컨테이너 내에서 GPU를 사용하려면 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)를 설치하세요.
* 코드 또는 아티팩트 소스 작업에서 이미지를 빌드하는 경우 [에이전트]({{< relref path="#configure-a-launch-agent-on-a-local-machine" lang="ko" >}})에서 사용하는 기본 이미지를 재정의하여 NVIDIA Container Toolkit을 포함할 수 있습니다.
  예를 들어, Launch 대기열 내에서 기본 이미지를 `tensorflow/tensorflow:latest-gpu`로 재정의할 수 있습니다.

  ```json
  {
    "builder": {
      "accelerator": {
        "base_image": "tensorflow/tensorflow:latest-gpu"
      }
    }
  }
  ```
{{% /alert %}}

## 대기열 생성

W&B CLI를 사용하여 Docker를 컴퓨팅 리소스로 사용하는 대기열을 만듭니다.

1. [Launch 페이지](https://wandb.ai/launch)로 이동합니다.
2. **Create Queue** 버튼을 클릭합니다.
3. 대기열을 만들려는 **Entity**를 선택합니다.
4. **Name** 필드에 대기열 이름을 입력합니다.
5. **Resource**로 **Docker**를 선택합니다.
6. **Configuration** 필드에 Docker 대기열 설정을 정의합니다.
7. **Create Queue** 버튼을 클릭하여 대기열을 만듭니다.

## 로컬 장치에서 Launch 에이전트 구성

`launch-config.yaml`이라는 YAML 구성 파일로 Launch 에이전트를 구성합니다. 기본적으로 W&B는 `~/.config/wandb/launch-config.yaml`에서 구성 파일을 확인합니다. Launch 에이전트를 활성화할 때 다른 디렉토리를 선택적으로 지정할 수 있습니다.

{{% alert %}}
W&B CLI를 사용하여 구성 YAML 파일 대신 Launch 에이전트에 대한 핵심 구성 가능 옵션(최대 작업 수, W&B Entity 및 Launch 대기열)을 지정할 수 있습니다. 자세한 내용은 [`wandb launch-agent`]({{< relref path="/ref/cli/wandb-launch-agent.md" lang="ko" >}}) 코맨드를 참조하세요.
{{% /alert %}}

## 핵심 에이전트 구성 옵션

다음 탭은 W&B CLI와 YAML 구성 파일로 핵심 구성 에이전트 옵션을 지정하는 방법을 보여줍니다.

{{< tabpane text=true >}}
{{% tab "W&B CLI" %}}
```bash
wandb launch-agent -q <queue-name> --max-jobs <n>
```
{{% /tab %}}

{{% tab "Config file" %}}
```yaml title="launch-config.yaml"
max_jobs: <n concurrent jobs>
queues:
	- <queue-name>
```
{{% /tab %}}
{{< /tabpane >}}

## Docker 이미지 빌더

장치의 Launch 에이전트를 구성하여 Docker 이미지를 빌드할 수 있습니다. 기본적으로 이러한 이미지는 장치의 로컬 이미지 리포지토리에 저장됩니다. Launch 에이전트가 Docker 이미지를 빌드할 수 있도록 하려면 Launch 에이전트 구성에서 `builder` 키를 `docker`로 설정합니다.

```yaml title="launch-config.yaml"
builder:
	type: docker
```

에이전트가 Docker 이미지를 빌드하지 않고 레지스트리에서 미리 빌드된 이미지를 대신 사용하려면 Launch 에이전트 구성에서 `builder` 키를 `noop`로 설정합니다.

```yaml title="launch-config.yaml"
builder:
  type: noop
```

## 컨테이너 레지스트리

Launch는 Dockerhub, Google Container Registry, Azure Container Registry 및 Amazon ECR과 같은 외부 컨테이너 레지스트리를 사용합니다.
빌드한 환경과 다른 환경에서 작업을 실행하려면 컨테이너 레지스트리에서 풀할 수 있도록 에이전트를 구성합니다.

Launch 에이전트를 클라우드 레지스트리에 연결하는 방법에 대한 자세한 내용은 [고급 에이전트 설정]({{< relref path="./setup-agent-advanced.md#agent-configuration" lang="ko" >}}) 페이지를 참조하세요.
