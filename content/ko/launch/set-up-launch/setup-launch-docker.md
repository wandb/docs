---
title: '튜토리얼: Docker로 W&B Launch 설정하기'
menu:
  launch:
    identifier: ko-launch-set-up-launch-setup-launch-docker
    parent: set-up-launch
url: guides/launch/setup-launch-docker
---

다음 가이드는 W&B Launch 를 로컬 머신에서 Docker 를 사용하도록 설정하는 방법을 설명합니다. 이는 launch agent 환경과 queue 의 target 리소스 모두에 적용됩니다.

같은 로컬 머신에서 Docker 를 job 실행과 launch agent 환경에 모두 사용하는 것은, 쿠버네티스(Kubernetes)와 같은 클러스터 관리 시스템이 없는 머신에 컴퓨팅 환경을 설치한 경우에 특히 유용합니다.

또한 Docker queue 를 활용하면 강력한 워크스테이션에서 워크로드를 실행할 수 있습니다.

{{% alert %}}
이 방식은 주로 자신의 로컬 머신에서 실험을 수행하거나, 원격 머신에 SSH 접속해 launch job 을 제출하는 사용자에게 일반적입니다.
{{% /alert %}}

Docker 를 W&B Launch 와 함께 사용하면, 먼저 이미지를 빌드하고, 그 이미지에서 컨테이너를 빌드·실행합니다. 이미지는 Docker 의 `docker run <image-uri>` 명령어로 빌드합니다. queue 설정 값들은 `docker run` 명령에 추가 인수로 해석되어 전달됩니다.


## Docker queue 설정하기

Launch queue 설정(Docker target 리소스용)은 [`docker run`]({{< relref path="/ref/cli/wandb-docker-run.md" lang="ko" >}}) CLI 명령에서 정의된 것과 동일한 옵션을 허용합니다.

에이전트는 queue 설정에서 정의된 옵션을 받습니다. 에이전트는 받은 옵션을 launch job 설정에서 오버라이드된 값과 병합해, 최종적으로 실행될 `docker run` 명령을 만듭니다(이 예시에서는 로컬 머신에서).

두 가지 문법 변환이 일어납니다:

1. 반복 옵션은 queue 설정에서 리스트로 정의합니다.
2. flag 옵션은 queue 설정에서 Boolean 타입의 `true` 값으로 정의합니다.

예를 들어, 다음과 같은 queue 설정은:

```json
{
  "env": ["MY_ENV_VAR=value", "MY_EXISTING_ENV_VAR"],
  "volume": "/mnt/datasets:/mnt/datasets",
  "rm": true,
  "gpus": "all"
}
```

아래와 같은 `docker run` 명령으로 변환됩니다:

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri> \
  --gpus all
```

볼륨은 문자열 리스트나 단일 문자열로 지정할 수 있습니다. 여러 개의 볼륨을 지정할 경우 리스트를 사용하세요.

Docker 는 값이 할당되지 않은 환경 변수를 launch agent 환경에서 자동으로 넘겨받습니다. 즉, launch agent 에 환경 변수 `MY_EXISTING_ENV_VAR`가 있다면 해당 환경 변수는 컨테이너에서도 사용할 수 있습니다. 이를 통해 queue 설정에 공개하지 않고 다른 설정 키를 활용할 수 있습니다.

`docker run` 명령의 `--gpus` 플래그는 Docker 컨테이너에서 사용할 GPU 를 지정할 수 있습니다. `gpus` 옵션 사용에 대한 자세한 내용은 [Docker 공식 문서](https://docs.docker.com/config/containers/resource_constraints/#gpu)를 참고하세요.

{{% alert %}}
* Docker 컨테이너 내에서 GPU 를 사용하려면 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)을 설치하세요.
* 코드 소스 또는 아티팩트 소스 job 에서 이미지를 빌드할 경우, [agent]({{< relref path="#configure-a-launch-agent-on-a-local-machine" lang="ko" >}})의 base image 를 NVIDIA Container Toolkit 이 포함된 이미지로 오버라이드할 수 있습니다.
  예를 들어, launch queue 내에서 아래와 같이 base image 를 `tensorflow/tensorflow:latest-gpu`로 지정할 수 있습니다:

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


## 큐 생성하기

W&B CLI 를 사용해 Docker 를 컴퓨팅 리소스로 사용하는 큐를 만드세요:

1. [Launch 페이지](https://wandb.ai/launch)로 이동합니다.
2. **Create Queue** 버튼을 클릭합니다.
3. 큐를 생성할 **Entity** 를 선택합니다.
4. **Name** 필드에 큐의 이름을 입력합니다.
5. **Resource** 로 **Docker** 를 선택합니다.
6. **Configuration** 필드에 Docker queue 설정을 입력합니다.
7. **Create Queue** 버튼을 클릭해 큐 생성 완료합니다.

## 로컬 머신에 launch agent 설정하기

`launch-config.yaml`이라는 YAML 파일로 launch agent 를 설정합니다. 기본적으로 W&B 는 `~/.config/wandb/launch-config.yaml` 경로에서 설정 파일을 찾습니다. launch agent 를 활성화할 때 다른 디렉터리를 지정할 수도 있습니다.

{{% alert %}}
W&B CLI 를 사용해(설정 YAML 파일 대신) launch agent 의 주요 옵션(최대 job 수, W&B 엔티티, launch queue 등)을 지정할 수 있습니다. 자세한 내용은 [`wandb launch-agent`]({{< relref path="/ref/cli/wandb-launch-agent.md" lang="ko" >}}) 명령어 설명을 참고하세요.
{{% /alert %}}


## Core agent 설정 옵션

아래 탭에서 W&B CLI 와 YAML 설정 파일로 core agent 옵션을 지정하는 방법을 볼 수 있습니다:

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

내 머신의 launch agent 는 Docker 이미지를 빌드하도록 설정할 수 있습니다. 기본적으로 이 이미지는 머신의 로컬 이미지 저장소에 저장됩니다. launch agent 가 Docker 이미지를 빌드하도록 활성화하려면, agent 설정에서 `builder` 키를 `docker`로 지정하세요:

```yaml title="launch-config.yaml"
builder:
	type: docker
```

에이전트가 Docker 이미지를 빌드하지 않고, 미리 빌드된 이미지를 registry 에서 사용하고 싶다면 agent 설정파일에서 `builder` 키를 `noop`으로 지정하세요.

```yaml title="launch-config.yaml"
builder:
  type: noop
```

## 컨테이너 레지스트리

Launch 는 Dockerhub, Google Container Registry, Azure Container Registry, Amazon ECR 등 외부 컨테이너 레지스트리를 사용합니다.  
이미지 빌드 환경과 다른 환경에서 job 을 실행하고 싶다면, agent 가 컨테이너 레지스트리에서 이미지를 pull 할 수 있도록 설정해야 합니다.

launch agent 를 클라우드 레지스트리와 연결하는 방법에 대한 자세한 내용은 [고급 agent 설정]({{< relref path="./setup-agent-advanced.md#agent-configuration" lang="ko" >}}) 페이지를 참고하세요.