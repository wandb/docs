---
description: Answers to frequently asked question about W&B Launch.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Launch 자주 묻는 질문

<head>
  <title>Launch에 관한 자주 묻는 질문</title>
</head>

## 시작하기

### W&B가 제게 컨테이너를 빌드하지 않기를 원합니다. 그래도 Launch를 사용할 수 있나요?

네. 다음 명령을 실행하여 사전 빌드된 도커 이미지를 실행하세요. `<>` 안의 항목을 귀하의 정보로 교체하세요:

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```  

이 명령은 실행을 생성할 때 작업을 빌드합니다.

또는 이미지에서 작업을 생성할 수 있습니다:

```bash
wandb job create image <image-name> -p <project> -e <entity>
```

### Launch를 효과적으로 사용하기 위한 모범 사례가 있나요?

  1. 에이전트를 시작하기 전에 큐를 생성하여 에이전트가 쉽게 지정할 수 있도록 합니다. 이 작업을 하지 않으면 에이전트가 오류를 발생시키고 큐를 추가할 때까지 작동하지 않습니다.
  2. W&B 서비스 계정을 사용하여 에이전트를 시작하면 개별 사용자 계정에 연결되지 않습니다.
  3. `wandb.config`를 사용하여 하이퍼파라미터를 읽고 쓰면 작업을 다시 실행할 때 덮어쓸 수 있습니다. argsparse를 사용하는 경우 [이 가이드](https://docs.wandb.ai/guides/launch/create-launch-job#making-your-code-job-friendly)를 확인하세요.

### 클릭하는 것이 싫습니다. UI를 거치지 않고도 Launch를 사용할 수 있나요?

  네. 표준 `wandb` CLI에는 작업을 실행할 수 있는 `launch` 하위 명령이 포함되어 있습니다. 자세한 정보는 다음을 실행해 보세요.

  ```bash
  wandb launch --help
  ```

### Launch가 대상 환경에서 컴퓨팅 리소스를 자동으로 프로비저닝(및 제거)할 수 있나요?

환경에 따라 다릅니다. SageMaker와 Vertex에서 리소스를 프로비저닝할 수 있습니다. Kubernetes에서는 필요할 때 리소스를 자동으로 확장하고 축소하는 데 사용할 수 있는 오토스케일러가 있습니다. W&B의 솔루션 아키텍트는 재시도, 오토스케일링 및 스팟 인스턴스 노드 풀 사용을 용이하게 하는 기본 Kubernetes 인프라를 구성하는 데 협력할 수 있습니다. support@wandb.com 또는 공유된 Slack 채널로 문의하세요.

### `wandb launch -d` 또는 `wandb job create image`가 도커 아티팩트 전체를 업로드하는 것이 아니라 레지스트리에서 가져오는 것인가요?

아닙니다. `wandb launch -d` 명령은 레지스트리에 대신 업로드하지 않습니다. 이미지를 직접 레지스트리에 업로드해야 합니다. 일반적인 단계는 다음과 같습니다:

1. 이미지를 빌드합니다.
2. 이미지를 레지스트리에 푸시합니다.

워크플로는 다음과 같습니다:

```bash
docker build -t <repo-url>:<tag> .
docker push <repo-url>:<tag>
wandb launch -d <repo-url>:<tag>
```

여기서부터, 런치 에이전트는 해당 컨테이너를 가리키는 작업을 시작합니다. 에이전트에게 컨테이너 레지스트리에서 이미지를 가져올 수 있는 권한을 부여하는 방법의 예는 [고급 에이전트 설정](./setup-agent-advanced.md#agent-configuration)을 참조하세요.

Kubernetes의 경우, Kubernetes 클러스터 포드는 푸시하는 레지스트리에 액세스할 수 있어야 합니다.

### Dockerfile을 지정하고 W&B가 Docker 이미지를 빌드하도록 할 수 있나요?
네. 요구 사항이 자주 변경되지 않지만 코드베이스가 자주 변경되는 경우 특히 유용합니다.

:::important
Dockerfile이 마운트를 사용하도록 포맷되었는지 확인하세요. 자세한 정보는 [Docker 문서 웹사이트의 마운트 문서](https://docs.docker.com/build/guide/mounts/)를 참조하세요.
:::

Dockerfile을 구성하면 W&B에 Dockerfile을 세 가지 방법 중 하나로 지정할 수 있습니다:

* Dockerfile.wandb 사용
* W&B CLI
* W&B 앱

<Tabs
  defaultValue="dockerfile"
  values={[
    {label: 'Dockerfile.wandb', value: 'dockerfile'},
    {label: 'W&B CLI', value: 'cli'},
    {label: 'W&B 앱', value: 'app'},
  ]}>
  <TabItem value="dockerfile">

W&B 실행의 진입점과 동일한 디렉터리에 `Dockerfile.wandb`라는 파일을 포함하세요. W&B는 W&B의 내장 Dockerfile 대신 `Dockerfile.wandb`를 사용합니다.

  </TabItem>
  <TabItem value="cli">

[`wandb launch`](../../ref/cli/wandb-launch.md) 명령을 호출할 때 `--dockerfile` 플래그를 제공하여 런치 작업을 큐에 추가하세요:

```bash
wandb launch --dockerfile path/to/Dockerfile
```

  </TabItem>
  <TabItem value="app">

W&B 앱에서 큐에 작업을 추가할 때 **Overrides** 섹션에 Dockerfile의 경로를 제공하세요. 구체적으로, `"dockerfile"`을 키로, 값은 Dockerfile의 경로로 하는 키-값 쌍을 제공합니다.

예를 들어, 다음 JSON은 로컬 디렉터리 내에 있는 Dockerfile을 포함하는 방법을 보여줍니다:

```json title="W&B 앱에서 런치 작업"
{
  "args": [],
  "run_config": {
    "lr": 0,
    "batch_size": 0,
    "epochs": 0
  },
  "entrypoint": [],
  "dockerfile": "./Dockerfile"
}
```

  </TabItem>
</Tabs>

## 권한 및 리소스

### 큐에 푸시할 수 있는 사람을 어떻게 제어하나요?

큐는 사용자 팀에 한정됩니다. 큐를 생성할 때 소유 엔터티를 정의합니다. 액세스를 제한하려면 팀 멤버십을 변경하세요.

### Kubernetes에서 에이전트에 필요한 권한은 무엇인가요?
“다음 쿠버네티스 매니페스트는 `wandb` 네임스페이스에서 `wandb-launch-agent`라는 이름의 역할을 생성합니다. 이 역할은 에이전트가 `wandb` 네임스페이스에서 파드, 컨피그맵, 시크릿, 및 파드/로그를 생성할 수 있도록 허용합니다. `wandb-cluster-role`은 에이전트가 선택한 모든 네임스페이스에서 파드, 파드/로그, 시크릿, 작업, 및 작업/상태를 생성할 수 있도록 허용합니다.”

### Launch는 병렬화를 지원하나요? 작업이 소비하는 리소스를 어떻게 제한할 수 있나요?

네, Launch는 여러 GPU와 여러 노드에 걸쳐 작업을 확장하는 것을 지원합니다. 자세한 내용은 [이 가이드](https://docs.wandb.ai/tutorials/volcano)를 참조하세요.

작업 간 수준에서, 개별 런치 에이전트는 에이전트가 동시에 실행할 수 있는 작업 수를 결정하는 `max_jobs` 파라미터로 구성됩니다. 또한, 해당 에이전트가 실행할 수 있는 인프라에 연결된 한, 특정 큐를 가리키는 많은 에이전트를 원하는 만큼 가리킬 수 있습니다.

런치 큐 또는 작업 실행 수준에서 CPU/GPU, 메모리 및 기타 요구 사항을 리소스 구성에서 제한할 수 있습니다. Kubernetes에서 리소스 제한이 있는 큐를 설정하는 방법에 대한 자세한 정보는 [여기](https://docs.wandb.ai/guides/launch/kubernetes#queue-configuration)를 참조하세요.

스윕의 경우, SDK에서 큐 구성에 블록을 추가할 수 있습니다

```yaml title="큐 구성"
  scheduler:
    num_workers: 4
```
스윕에서 동시에 실행될 작업 수를 제한합니다.

### Docker 큐를 사용하여 `use_artifact`로 동일한 아티팩트를 다운로드하는 여러 작업을 실행할 때, 작업 실행마다 아티팩트를 다시 다운로드하나요, 아니면 백그라운드에서 캐싱이 이루어지나요?

캐싱이 없습니다; 각 작업은 독립적입니다. 그러나 큐/에이전트를 공유 캐시를 마운트하는 방식으로 구성할 수 있는 방법이 있습니다. 큐 구성에서 도커 인수를 통해 이를 달성할 수 있습니다.

특별한 경우, W&B 아티팩트 캐시를 지속적인 볼륨으로 마운트할 수도 있습니다.

### 작업/자동화에 대한 비밀을 지정할 수 있나요? 예를 들어, 사용자가 직접 볼 수 없는 API 키와 같은 것입니다.

네. 권장하는 방법은 다음과 같습니다:

  1. 실행이 생성될 네임스페이스에 일반 k8s 시크릿으로 비밀을 추가합니다. 예를 들어 `kubectl create secret -n <namespace> generic <secret_name> <secret value>`

  2. 해당 비밀이 생성되면, 실행이 시작될 때 비밀을 주입하도록 큐 구성을 지정할 수 있습니다. 최종 사용자는 비밀을 볼 수 없으며, 클러스터 관리자만 볼 수 있습니다.

### 관리자가 ML 엔지니어가 수정할 수 있는 것을 어떻게 제한할 수 있나요? 예를 들어, 이미지 태그를 변경하는 것은 괜찮지만 다른 작업 설정은 그렇지 않을 수 있습니다.

[큐 구성 템플릿](./setup-queue-advanced.md)을 통해 제어할 수 있으며, 관리자 사용자가 정의한 제한 내에서 비팀 관리 사용자가 편집할 수 있는 특정 큐필드를 노출합니다. 팀 관리자만 큐를 생성하거나 편집할 수 있으며, 노출되는 필드와 해당 필드의 제한을 정의할 수 있습니다.

### W&B Launch는 어떻게 이미지를 빌드하나요?

이미지 빌드에 수행되는 단계는 실행되는 작업의 출처와 리소스 구성에 지정된 가속기 기반 이미지 여부에 따라 달라집니다.

:::note
큐 구성을 지정하거나 작업을 제출할 때, 큐 또는 작업 리소스 구성에서 가속기 기반 이미지를 제공할 수 있습니다:
```json
{
    "builder": {
        "accelerator": {
            "base_image": "image-name"
        }
    }
}
```
:::

빌드 프로세스 중에는 제공된 가속기 기반 이미지와 작업 유형에 따라 다음 작업이 수행됩니다:

|                                                     | apt를 사용하여 python 설치 | python 패키지 설치 | 사용자 및 작업 디렉터리 생성 | 코드를 이미지에 복사 | 진입점 설정 |
|-----------------------------------------------------|:------------------------:|:-----------------------:|:-------------------------:|:--------------------:|:--------------:|
| git에서 출처한 작업                                |                          |            X            |             X             |           X          |        X       |
| 코드에서 출처한 작업                               |                          |            X            |             X             |           X          |        X       |
| git에서 출처한 작업 및 제공된 가속기 이미지 |             X            |            X            |             X             |           X          |        X       |
| 코드에서 출처한 작업 및 제공된 가속기 이미지|             X            |            X            |             X             |           X          |        X       |
| 이미지에서 출처한 작업                              |                          |                         |                           |                      |                |

### 가속기 기반 이미지에 필요한 요구 사항은 무엇인가요?
가속기를 사용하는 작업의 경우, 필요한 가속기 구성 요소가 설치된 가속기 기반 이미지를 제공할 수 있습니다. 제공된 가속기 이미지에 대한 기타 요구 사항은 다음과 같습니다:
- Debian 호환성 (Launch Dockerfile은 python을 가져오기 위해 apt-get을 사용합니다)
- CPU 및 GPU 하드웨어 명령 세트 호환성 (사용하려는 GPU에서 지원하는 CUDA 버전인지 확인하세요)
- 제공하는 가속기 버전과 ML 알고리즘에 설치된 패키지 간의 호환성
- 하드웨어와의 호환성을 설정하기 위한 추가 단계가 필요한 패키지 설치

### GPU에서 Tensorflow를 사용하는 W&B Launch를 어떻게 작동시키나요?
GPU에서 tensorflow를 사용하는 작업의 경우, 에이전트가 실행을 제대로 GPU를 사용할 수 있도록 컨테이너 빌드에 사용할 사용자 지정 기반 이미지를 지정해야 할 수도 있습니다. 이는 리소스 구성에서 `builder.accelerator.base_image` 키 아래에 이미지 태그를 추가하여 수행할 수 있습니다. 예를 들어:

```json
{
    "gpus": "all",
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```

wandb 버전 0.15.6 이전에는 `base_image`의 상위 키로 `cuda` 대신 `accelerator`를 사용하세요.

### Launch가 이미지를 빌드할 때 사용자 지정 저장소를 사용할 수 있나요?

네. 가능합니다. `requirements.txt`에 다음 줄을 추가하고 `index-url` 및 `extra-index-url`에 전달된 값을 자신의 값으로 교체하세요:

```text
----index-url=https://xyz@<your-repo-host> --extra-index-url=https://pypi.org/simple
```

`requirements.txt`는 작업의 기본 루트에 정의되어야 합니다.

## 사전 중지된 노드에서의 자동 실행 재큐잉

일부 경우, 중단된 작업을 이어서 실행하는 것이 유용할 수 있