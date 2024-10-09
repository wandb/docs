---
title: Launch FAQ
description: W&B Launch에 대한 자주 묻는 질문에 대한 답변.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## 시작하기

### W&B가 저를 위해 컨테이너를 만들어 주기를 원하지 않는데, Launch를 여전히 사용할 수 있나요? 

네. 미리 빌드된 도커 이미지를 실행하려면 다음을 실행하세요. `<>`의 항목을 귀하의 정보로 대체하세요:

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```

이렇게 하면 run을 만들 때 작업이 생성됩니다.

또는 이미지를 사용하여 작업을 만들 수 있습니다:

```bash
wandb job create image <image-name> -p <project> -e <entity>
```

### Launch를 효과적으로 사용하는 모범 사례가 있나요?

  1. 에이전트를 시작하기 전에 큐를 생성하여 에이전트가 쉽게 가리킬 수 있도록 설정하세요. 이렇게 하지 않으면 에이전트가 오류를 발생시키고 큐가 추가될 때까지 작동하지 않습니다.
  2. 에이전트를 시작하기 위해 W&B 서비스 계정을 생성하세요. 개인 사용자 계정에 묶이지 않도록 합니다.
  3. `wandb.config`를 사용하여 하이퍼파라미터를 읽고 쓰도록 하세요. 이렇게 하면 작업을 다시 실행할 때 덮어쓸 수 있습니다. argparse를 사용하는 경우 [이 가이드](/guides/track/config/#set-the-configuration-with-argparse)를 참조하세요.

### 클릭하기 싫은데 UI를 거치지 않고 Launch를 사용할 수 있나요? 

예. 표준 `wandb` CLI에는 작업을 시작할 수 있는 `launch` 하위 명령어가 포함되어 있습니다. 자세한 내용을 보려면 다음을 실행해 보세요
 
```bash
wandb launch --help
```

### Launch가 타겟 환경에서 컴퓨트 리소스를 자동으로 제공하고 종료할 수 있나요?

환경에 따라 다릅니다. SageMaker와 Vertex에서 리소스를 제공할 수 있습니다. Kubernetes에서는 자동 조정기가 필요할 때 리소스를 자동으로 시작하고 종료하도록 사용할 수 있습니다. W&B의 솔루션 아키텍트가 재시도, 자동 확장 및 스팟 인스턴스 노드 풀 사용을 지원하기 위해 기본 Kubernetes 인프라를 구성하는 데 기꺼이 협력할 것입니다. support@wandb.com 또는 사용 중인 Slack 채널로 문의하세요.

### `wandb launch -d` 또는 `wandb job create image`가 전체 도커 아티팩트를 업로드하고 레지스트리에서 가져오지 않는 것인가요?

아니요. `wandb launch -d` 명령은 레지스트리에 업로드하지 않습니다. 이미지를 직접 레지스트리에 업로드해야 합니다. 다음은 일반적인 단계입니다:

1. 이미지를 빌드합니다.
2. 이미지를 레지스트리에 푸시합니다.

워크플로우는 다음과 같이 진행됩니다:

```bash
docker build -t <repo-url>:<tag> .
docker push <repo-url>:<tag>
wandb launch -d <repo-url>:<tag>
```

거기에서 출시 에이전트가 해당 컨테이너를 가리키는 작업을 시작합니다. 컨테이너 레지스트리에서 이미지를 가져오는 것에 대한 에이전트 엑세스를 제공하는 방법의 예를 보려면 [고급 에이전트 설정](./setup-agent-advanced.md#agent-configuration)을 참조하세요.

Kubernetes의 경우, Kubernetes 클러스터 포드가 푸시할 레지스트리에 엑세스해야 합니다.

### Dockerfile을 지정하고 W&B가 도커 이미지를 빌드하게 할 수 있나요?

네. 변경사항이 자주 발생하지 않는 많은 요구사항이 있지만 코드베이스는 자주 변경되는 경우에 특히 유용합니다.

:::important
Dockerfile이 마운트를 사용하도록 형식이 지정되었는지 확인하세요. 자세한 내용은 [Docker Docs 웹사이트에서 마운트 문서](https://docs.docker.com/build/guide/mounts/)를 참조하세요.
:::

Dockerfile이 설정되면 다음 중 하나의 방법으로 W&B에 Dockerfile을 지정할 수 있습니다:

- Dockerfile.wandb 사용
- W&B CLI
- W&B App


<Tabs
  defaultValue="dockerfile"
  values={[
    {label: 'Dockerfile.wandb', value: 'dockerfile'},
    {label: 'W&B CLI', value: 'cli'},
    {label: 'W&B App', value: 'app'},
  ]}>
  <TabItem value="dockerfile">

W&B run의 엔트리포인트와 같은 디렉토리에 `Dockerfile.wandb`라는 파일을 포함하십시오. W&B는 자체 Dockerfile 대신 `Dockerfile.wandb`를 사용합니다.


  </TabItem>
  <TabItem value="cli">

[`wandb launch`](../../ref/cli/wandb-launch.md) 명령으로 작업 실행을 대기열에 넣을 때 `--dockerfile` 플래그를 제공하십시오:

```bash
wandb launch --dockerfile path/to/Dockerfile
```


  </TabItem>
  <TabItem value="app">


W&B 앱에서 큐에 작업을 추가할 때 **수정** 섹션에 Dockerfile의 경로를 제공합니다. 좀 더 구체적으로 말하면, 그것을 "dockerfile"이 키이고 값이 Dockerfile의 경로인 키-값 쌍으로 제공합니다.

예를 들어, 다음 JSON은 로컬 디렉토리에 있는 Dockerfile을 포함하는 방법을 보여줍니다:

```json title="Launch job W&B App"
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

### 누가 큐에 푸시할 수 있는지 어떻게 제어하나요?

큐는 사용자 팀에 범위가 지정됩니다. 큐를 생성할 때 소유 엔티티를 정의합니다. 엑세스를 제한하려면 팀 멤버십을 변경할 수 있습니다.

### Kubernetes에서 에이전트가 필요한 권한은 무엇입니까?
다음 Kubernetes 매니페스트는 `wandb` 네임스페이스에 `wandb-launch-agent`라는 역할을 생성합니다. 이 역할을 통해 에이전트는 `wandb` 네임스페이스에서 포드, configmaps, 비밀 및 pods/log를 생성할 수 있습니다. `wandb-cluster-role`을 사용하면 에이전트가 선택한 네임스페이스에서 포드, pods/log, 비밀, 작업 및 작업/상태를 생성할 수 있습니다.

### Launch가 병렬 처리를 지원하나요? 작업에서 소비하는 리소스를 어떻게 제한할 수 있나요?

네, Launch는 여러 GPU와 여러 노드에 걸쳐 작업을 확장하는 것을 지원합니다. 자세한 내용은 [이 가이드](/tutorials/volcano)를 참조하세요.

작업 간 수준에서, 개별 launch 에이전트는 에이전트가 동시에 실행할 수 있는 작업 수를 결정하는 `max_jobs` 파라미터로 설정됩니다. 추가적으로, 해당 에이전트들이 실행할 수 있는 인프라에 연결되어 있는 한 특정 큐에 원하는 만큼 많은 에이전트를 지정할 수 있습니다.

CPU/GPU, 메모리 및 기타 요구 사항은 리소스 설정에서 launch 큐 또는 작업 실행 수준에서 제한할 수 있습니다. Kubernetes에서 리소스 제한이 있는 큐를 설정하는 방법에 대한 자세한 내용은 [여기](setup-launch-kubernetes)를 참조하세요.

Sweeps의 경우, SDK에서 큐 설정에 블록을 추가할 수 있습니다:

```yaml title="queue config"
  scheduler:
    num_workers: 4
```
이렇게 하면 병렬로 실행될 스윕의 동시 실행 횟수가 제한됩니다.

### `use_artifact`로 동일한 아티팩트를 다운로드하는 여러 작업을 실행하는 Docker 큐를 사용할 때, 작업의 매번 실행 시 아티팩트를 다시 다운로드하나요, 아니면 백그라운드에서 캐시가 있나요?

캐시는 없습니다. 각 작업은 독립적입니다. 그러나 큐/에이전트를 설정하여 공유 캐시를 마운트하는 방법이 있습니다. 큐 설정에서 도커 인수를 통해 이를 달성할 수 있습니다.

특수한 경우로, W&B 아티팩트 캐시를 영구 볼륨으로 마운트할 수도 있습니다.

### 작업/자동화를 위한 비밀을 지정할 수 있나요? 예를 들어, 사용자에게 직접 보이는 것을 원하지 않는 API 키를?

네. 권장 방법은 다음과 같습니다:

  1. 실행이 생성될 네임스페이스에 일반적인 k8s 비밀로 비밀을 추가합니다. 예를 들어 `kubectl create secret -n <namespace> generic <secret_name> <secret value>` 같은 것이 있습니다.

 2. 그 비밀이 생성되고 나면, 실행이 시작될 때 비밀을 주입하도록 큐 설정을 지정할 수 있습니다. 최종 사용자는 비밀을 볼 수 없으며, 클러스터 관리자만 볼 수 있습니다.

### ML 엔지니어가 수정할 수 있는 항목을 관리자가 어떻게 제한할 수 있나요? 예를 들어, 이미지 태그를 변경하는 것은 괜찮지만 다른 작업 설정은 아닐 수 있습니다.

이는 관리 사용자가 정의한 제한 내에서 비 팀 관리자 사용자가 특정 큐 필드를 수정할 수 있도록 하는 [큐 설정 템플릿](./setup-queue-advanced.md)에 의해 제어될 수 있습니다. 팀 관리자만 큐를 생성하거나 편집할 수 있으며, 노출된 필드 및 제한 사항을 정의할 수 있습니다.

### W&B Launch는 이미지를 어떻게 빌드하나요?

이미지를 빌드하기 위한 단계는 실행되는 작업의 소스와 리소스 설정이 가속기 기반 이미지를 지정하는지 여부에 따라 다릅니다.

:::note
큐 설정을 지정하거나 작업을 제출할 때, 큐 또는 작업 리소스 설정에서 기본 가속기 이미지를 제공할 수 있습니다:
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

빌드 과정에서 제공된 작업 및 가속기 기본 이미지 유형에 따라 다음 작업이 수행됩니다:

|                                                     | apt로 python 설치 | python 패키지 설치 | 사용자 및 작업 디렉토리 생성 | 이미지를 코드로 복사 | 엔트리포인트 설정 |
|-----------------------------------------------------|:-----------------:|:-----------------:|:---------------------:|:----------------:|:--------------:|
| 깃에서 가져온 작업                                   |                    |         X         |           X           |     X            |       X      |
| 코드에서 가져온 작업                                  |                    |         X         |           X           |     X            |       X      |
| 깃에서 가져온 작업 및 가속기 이미지 제공              |         X         |         X         |           X           |     X            |       X      |
| 코드에서 가져온 작업 및 가속기 이미지 제공            |         X         |         X         |           X           |     X            |       X      |
| 이미지에서 가져온 작업                                |                   |                   |                           |                    |                |

### 가속기 기본 이미지의 요구 사항은 무엇인가요?
가속기를 사용하는 작업의 경우, 필요한 가속기 구성 요소가 설치된 가속기 기본 이미지를 제공할 수 있습니다. 제공된 가속기 이미지에 대한 다른 요구 사항은 다음과 같습니다:
- Debian 호환성 (Launch Dockerfile은 python을 가져오기 위해 apt-get을 사용합니다)
- CPU & GPU 하드웨어 명령 세트와의 호환성 (사용하려는 GPU에서 지원되는 CUDA 버전을 확인하세요)
- 제공한 가속기 버전과 ML 알고리즘에 설치된 패키지 간의 호환성
- 하드웨어와의 호환성을 설정하기 위해 추가 단계가 필요한 패키지 설치

### W&B Launch를 GPU에서 Tensorflow와 함께 작동하게 하려면 어떻게 해야 하나요?
GPU에서 Tensorflow를 사용하는 작업의 경우, 에이전트가 GPU를 적절히 활용할 수 있도록 실행할 컨테이너 빌드를 위한 사용자 정의 기본 이미지를 지정해야 할 수도 있습니다. 예를 들어, 리소스 설정에 `builder.accelerator.base_image` 키 아래에 이미지 태그를 추가하여 이를 수행할 수 있습니다:

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

wandb 버전 0.15.6 이전에는 `accelerator` 대신 `cuda`를 `base_image`의 상위 키로 사용하세요.

### Launch가 이미지를 빌드할 때 맞춤형 패키지 저장소를 사용할 수 있나요?

네. 그렇게 하려면, `requirements.txt` 파일에 다음 줄을 추가하고 `index-url` 및 `extra-index-url`에 전달된 값을 본인 것으로 바꿉니다:

```text
----index-url=https://xyz@<your-repo-host> --extra-index-url=https://pypi.org/simple
```

`requirements.txt`는 작업의 루트에 정의되어 있어야 합니다.

## 일시 정지 시 자동 run 다시 대기열 설정

일부 경우에는 중단된 후 작업을 재개하도록 설정하는 것이 유용할 수 있습니다. 예를 들어, 스팟 인스턴스에서 광범위한 하이퍼파라미터 탐색을 실행하고, 스팟 인스턴스가 다시 시작될 때 이를 다시 시작하도록 하고 싶을 수 있습니다. Launch는 Kubernetes 클러스터에서 이 구성을 지원할 수 있습니다.

Kubernetes 큐가 스케줄러에 의해 선점된 노드에서 작업을 실행 중인 경우, 작업은 대기열의 끝에 자동으로 다시 추가되어 나중에 재개될 수 있습니다. 이 재개된 run은 원래와 동일한 이름을 가지며, UI의 원래 페이지에서 계속 추적할 수 있습니다. 작업은 최대 다섯 번까지 자동으로 다시 대기열 설정될 수 있습니다.

Launch는 포드가 `DisruptionTarget` 상태를 가지는지 여부를 확인하여 포드가 스케줄러에 의해 선점되었는지를 감지합니다. 이유는 다음 중 하나일 수 있습니다:

- `EvictionByEvictionAPI`
- `PreemptionByScheduler`
- `TerminationByKubelet`

작업의 코드를 이어서 재개할 수 있도록 구조화하면 재대기열된 run들이 중단된 부분에서 다시 시작할 수 있습니다. 그렇지 않으면, 다시 대기열에 추가될 때마다 run들은 처음부터 시작하게 됩니다. [재개하는 run](../runs/resuming.md) 가이드에서 더 많은 정보를 확인하세요.

현재로서는, 선점된 노드에 대해 run을 자동으로 다시 대기열 설정을 해제할 방법이 없습니다. 그러나 UI에서 run을 삭제하거나 노드를 직접 삭제하면 다시 대기열에 추가되지 않습니다.

자동 run 다시 대기열 설정은 현재 Kubernetes 큐에서만 가능합니다. Sagemaker와 Vertex는 아직 지원되지 않습니다.