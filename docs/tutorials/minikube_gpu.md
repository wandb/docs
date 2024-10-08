---
title: Spin up a single node GPU cluster with Minikube
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B Launch를 설정하여 GPU 워크로드를 스케줄링하고 실행할 수 있는 Minikube 클러스터를 구성하세요.

:::안내
이 튜토리얼은 GPU를 여러 개 보유한 머신에 직접 엑세스할 수 있는 사용자를 위한 가이드입니다. 클라우드 머신을 임대하여 사용하는 사용자를 위한 것은 아닙니다.

클라우드 머신에 minikube 클러스터를 설정하려면, 클라우드 공급자를 사용하여 GPU 지원을 갖춘 Kubernetes 클러스터를 만들 것을 W&B에서 권장합니다. 예를 들어 AWS, GCP, Azure, Coreweave 등 클라우드 공급자는 GPU 지원 Kubernetes 클러스터를 만드는 툴을 제공하고 있습니다.

단일 GPU를 보유한 머신에서 GPU를 스케줄링하기 위해 minikube 클러스터를 설정하려는 경우, W&B는 [Launch Docker queue](/guides/launch/setup-launch-docker)를 사용할 것을 권장합니다. 여전히 재미로 튜토리얼을 따라갈 수 있지만, GPU 스케줄링은 크게 유용하지 않을 것입니다.
:::

## 배경

[Nvidia 컨테이너 툴킷](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) 덕분에 GPU가 지원되는 워크플로우를 Docker에서 쉽게 실행할 수 있습니다. 제한점은 볼륨에 따라 GPU를 스케줄링하는 네이티브 지원이 부족하다는 것입니다. `docker run` 코맨드에서 GPU를 사용하려면 특정 GPU ID를 요청하거나 모든 GPU를 요청해야 하므로 많은 분산 GPU 지원 워크로드가 비현실적입니다. Kubernetes는 볼륨 요청에 따라 스케줄링하는 지원을 제공하지만, GPU 스케줄링이 가능한 로컬 Kubernetes 클러스터를 설정하는 데는 상당한 시간과 노력이 필요했습니다. Minikube는 가장 인기 있는 단일 노드 Kubernetes 클러스터 실행 툴 중 하나로, 최근 [GPU 스케줄링 지원](https://minikube.sigs.k8s.io/docs/tutorials/nvidia/)을 출시했습니다 🎉 이 튜토리얼에서 우리는 Multi-GPU 머신에 Minikube 클러스터를 만들고, W&B Launch를 사용하여 클러스터에 동시 안정성 확산 추론 작업을 실행할 것입니다 🚀

## 필요 조건

시작하기 전에 다음이 필요합니다:

1. W&B 계정.
2. 다음을 설치 및 실행한 Linux 머신:
   1. Docker 런타임
   2. 사용할 모든 GPU에 대한 드라이버
   3. Nvidia 컨테이너 툴킷

:::note
이 튜토리얼을 테스트하고 작성할 때, 우리는 4개의 NVIDIA Tesla T4 GPU가 연결된 `n1-standard-16` Google Cloud Compute Engine 인스턴스를 사용했습니다.
:::

## Launch 작업을 위한 큐 생성

우선, Launch 작업을 위한 큐를 생성하세요.

1. [wandb.ai/launch](https://wandb.ai/launch) (혹은 프라이빗 W&B 서버를 사용하는 경우 `<your-wandb-url>/launch`)로 이동합니다.
2. 화면 오른쪽 상단 모서리에서 파란색 **큐 생성** 버튼을 클릭하세요. 큐 생성 서랍이 화면 오른쪽에서 슬라이드 아웃됩니다.
3. 엔터티를 선택하고 이름을 입력한 후, 큐의 유형으로 **Kubernetes**를 선택하세요.
4. 서랍의 **Config** 섹션에서는 Launch queue에 대한 [Kubernetes 작업 사양](https://kubernetes.io/docs/concepts/workloads/controllers/job/)을 입력합니다. 이 큐에서 시작된 모든 run은 이 작업 사양을 사용하여 생성되므로, 이를 수정하여 작업을 사용자 정의할 수 있습니다. 이 튜토리얼을 위해, 아래 샘플 설정을 YAML 또는 JSON 형식으로 큐 설정란에 복사하여 붙여 넣을 수 있습니다:

   

<Tabs
defaultValue="yaml"
values={[
{ label: "YAML", value: "yaml", },
{ label: "JSON", value: "json", },
]}>

<TabItem value="yaml">

spec:
  template:
    spec:
      containers:
        - image: ${image_uri}
          resources:
            limits:
              cpu: 4
              memory: 12Gi
              nvidia.com/gpu: '{{gpus}}'
      restartPolicy: Never
  backoffLimit: 0
  

</TabItem>

<TabItem value="json">

{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "image": "${image_uri}",
            "resources": {
              "limits": {
                "cpu": 4,
                "memory": "12Gi",
                "nvidia.com/gpu": "{{gpus}}"
              }
            }
          }
        ],
        "restartPolicy": "Never"
      }
    },
    "backoffLimit": 0
  }
}
  
</TabItem>
</Tabs>

큐 설정에 대한 추가 정보는 [Kubernetes에 Launch 설정하기](/guides/launch/setup-launch-kubernetes.md)와 [고급 큐 설정 가이드](/guides/launch/setup-queue-advanced.md)를 참조하세요.

`${image_uri}`와 `{{gpus}}` 문자열은 큐 설정에서 사용할 수 있는 두 가지 유형의 변수 템플릿 예시입니다. `${image_uri}` 템플릿은 런치할 작업의 이미지 URI로 에이전트에 의해 교체됩니다. `{{gpus}}` 템플릿은 작업 제출 시 Launch UI, CLI 또는 SDK에서 대체할 수 있는 템플릿 변수를 만들기 위해 사용됩니다. 이러한 값은 작업 사양에 배치되므로 작업에 사용되는 이미지와 GPU 자원을 제어하는 올바른 필드를 수정할 수 있습니다.

5. **구성 구문 분석** 버튼을 클릭하여 `gpus` 템플릿 변수를 사용자 정의하기 시작하세요.
6. **Type**을 `Integer`로 설정하고 **기본값**, **최소값**, **최대값**을 원하는 값으로 설정하세요.
템플릿 변수 제약 조건을 위반하여 이 큐에 run을 제출하려고 하면 거부됩니다.

![gpus 템플릿 변수가 있는 큐 생성 서랍 이미지](/images/tutorials/minikube_gpu/create_queue.png)

7. **큐 생성**을 클릭하여 큐를 생성하세요. 새 큐의 큐 페이지로 리디렉션됩니다.

다음 섹션에서는 생성한 큐에서 작업을 가져와 실행할 수 있는 에이전트를 설정할 것입니다.

## Docker + NVIDIA CTK 설정

Docker와 Nvidia 컨테이너 툴킷이 이미 머신에 설치되어 있다면 이 섹션을 건너뛰어도 됩니다.

Docker 컨테이너 엔진 설정에 대한 설명은 [Docker 문서](https://docs.docker.com/engine/install/)를 참조하세요.

Docker가 설치되면 [Nvidia 문서](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)의 지침에 따라 Nvidia 컨테이너 툴킷을 설치하세요.

컨테이너 런타임이 GPU 엑세스가 가능한지 확인하려면 다음을 실행하세요:

```bash
docker run --gpus all ubuntu nvidia-smi
```

이 장비에 연결된 GPU를 설명하는 `nvidia-smi` 출력이 표시되어야 합니다. 예를 들어, 우리의 설정에서 출력은 다음과 같습니다:

```
Wed Nov  8 23:25:53 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   38C    P8     9W /  70W |      2MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla T4            Off  | 00000000:00:05.0 Off |                    0 |
| N/A   38C    P8     9W /  70W |      2MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Tesla T4            Off  | 00000000:00:06.0 Off |                    0 |
| N/A   40C    P8     9W /  70W |      2MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Tesla T4            Off  | 00000000:00:07.0 Off |                    0 |
| N/A   39C    P8     9W /  70W |      2MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Minikube 설정

Minikube의 GPU 지원은 `v1.32.0` 버전 이상이 필요합니다. 최신 설치 도움말에 대한 [Minikube의 설치 문서](https://minikube.sigs.k8s.io/docs/start/)를 참조하세요. 이 튜토리얼에서는 다음 코맨드를 사용하여 최신 Minikube 릴리스를 설치했습니다:

```yaml
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

다음 단계는 GPU를 사용하여 minikube 클러스터를 시작하는 것입니다. 머신에서 다음을 실행하세요:

```yaml
minikube start --gpus all
```

위 코맨드의 출력은 클러스터가 성공적으로 생성되었는지 여부를 나타냅니다.

## Launch 에이전트 시작

새 클러스터의 Launch 에이전트는 `wandb launch-agent`를 직접 호출하거나 [W&B에서 관리하는 helm 차트](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)를 사용하여 에이전트를 배포하여 시작할 수 있습니다.

이 튜토리얼에서는 호스트 머신에서 직접 에이전트를 실행할 것입니다.

:::tip
컨테이너 외부에서 에이전트를 실행하면 로컬 Docker 호스트를 사용하여 클러스터가 실행할 이미지를 빌드할 수도 있습니다.
:::

에이전트를 로컬에서 실행하려면 기본 Kubernetes API 컨텍스트가 Minikube 클러스터를 참조하고 있는지 확인하세요. 그런 다음 다음을 실행하세요:

```bash
pip install "wandb[launch]"
```

에이전트의 종속 항목을 설치합니다. 에이전트 인증 설정을 위해 `wandb login`을 실행하거나 `WANDB_API_KEY` 환경 변수를 설정하세요.

에이전트를 시작하려면 다음을 입력하고 실행하세요:

```bash
wandb launch-agent -j <max-number-concurrent-jobs> -q <queue-name> -e <queue-entity>
```

터미널에서 에이전트가 run을 실행하고 시작하려는 메시지를 인쇄하기 시작하는 것을 보게 될 것입니다.

축하합니다! Launch 에이전트가 Launch queue를 폴링하고 있습니다! 큐에 작업이 추가되면 에이전트가 이를 선택하여 Minikube 클러스터에서 실행되도록 스케줄링할 것입니다.

## 작업 시작

에이전트로 작업을 보내봅시다. 터미널에서 W&B 계정에 로그인한 상태로 간단한 "hello world"를 시작할 수 있습니다:

```yaml
wandb launch -d wandb/job_hello_world:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

원하는 작업이나 이미지를 테스트할 수 있지만, 클러스터에서 이미지를 가져올 수 있는지 확인하세요. 추가 안내는 [Minikube의 문서](https://minikube.sigs.k8s.io/docs/handbook/registry/)를 참조하세요. 또한 [우리의 공개 작업 중 하나를 사용하여 테스트할 수 있습니다](https://wandb.ai/wandb/jobs/jobs?workspace=user-bcanfieldsherman).

## (선택적) 모델 및 데이터 캐싱을 위한 NFS

ML 워크로드에서 여러 작업이 동일한 데이터에 엑세스할 수 있도록 하고 싶을 때가 종종 있습니다. 예를 들어, 데이터셋이나 모델 가중치와 같은 대용량 자산을 반복적으로 다운로드하는 것을 피하기 위해 공유 캐시를 보유하고자 할 수 있습니다. Kubernetes는 [Persistent Volumes 및 Persistent Volume Claims](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)을 통해 이를 지원합니다. Persistent Volumes는 Kubernetes 워크로드에 `volumeMounts`를 생성하는 데 사용될 수 있으며, 공유 캐시에 대한 직접적 파일 시스템 엑세스를 제공합니다.

이 단계에서는 모델 가중치에 대한 공유 캐시로 사용할 수 있는 네트워크 파일 시스템(NFS) 서버를 설정할 것입니다. 첫 번째 단계는 NFS를 설치하고 구성하는 것입니다. 이 과정은 운영 체제에 따라 다릅니다. 우리의 VM은 Ubuntu를 실행 중이므로 nfs-kernel-server를 설치하고 `/srv/nfs/kubedata`에 export를 구성합니다:

```bash
sudo apt-get install nfs-kernel-server
sudo mkdir -p /srv/nfs/kubedata
sudo chown nobody:nogroup /srv/nfs/kubedata
sudo sh -c 'echo "/srv/nfs/kubedata *(rw,sync,no_subtree_check,no_root_squash,no_all_squash,insecure)" >> /etc/exports'
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

NFS 서버의 파일 시스템 내 export 위치와 로컬 IP 어드레스를 기록해 두십시오. 다음 단계에서 이 정보를 사용해야 합니다.

다음으로, 이 NFS에 대한 Persistent Volume 및 Persistent Volume Claim을 생성해야 합니다. Persistent Volumes는 매우 사용자 정의 가능합니다. 하지만 여기서는 간단한 구성을 사용할 것입니다.

아래 yaml을 `nfs-persistent-volume.yaml`이라는 이름의 파일에 복사하되, 원하는 볼륨 용량 및 클레임 요청을 작성하세요. `PersistentVolume.spec.capcity.storage` 필드는 기본 볼륨의 최대 크기를 제어합니다. `PersistentVolumeClaim.spec.resources.requests.stroage`는 특정 클레임에 할당된 볼륨 용량을 제한하는 데 사용할 수 있습니다. 우리의 유스 케이스에서는 두 값에 동일한 값을 사용하는 것이 합리적입니다.

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv
spec:
  capacity:
    storage: 100Gi # 원하는 용량으로 설정하십시오.
  accessModes:
    - ReadWriteMany
  nfs:
    server: <your-nfs-server-ip> # TODO: 여기에 작성하십시오.
    path: '/srv/nfs/kubedata' # 또는 사용자 정의 경로
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nfs-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi # 원하는 용량으로 설정하십시오.
  storageClassName: ''
  volumeName: nfs-pv
```

다음 명령어를 사용하여 클러스터에 자원을 생성하세요:

```yaml
kubectl apply -f nfs-persistent-volume.yaml
```

NFS를 캐시로 사용하려면, 런치 큐 설정에 `volumes` 및 `volumeMounts`를 추가해야 합니다. 런치 설정을 편집하려면 다시 [wandb.ai/launch](http://wandb.ai/launch) (혹은 wandb 서버의 경우 \<your-wandb-url\>/launch)로 이동하여 큐를 찾고, 큐 페이지로 가서 **구성 편집** 탭을 클릭하세요. 원래 설정을 다음과 같이 수정할 수 있습니다:

<Tabs
defaultValue="yaml"
values={[
{ label: "YAML", value: "yaml", },
{ label: "JSON", value: "json", },
]}>

<TabItem value="yaml">

spec:
  template:
    spec:
      containers:
        - image: ${image_uri}
          resources:
            limits:
              cpu: 4
              memory: 12Gi
              nvidia.com/gpu: "{{gpus}}"
					volumeMounts:
            - name: nfs-storage
              mountPath: /root/.cache
      restartPolicy: Never
			volumes:
        - name: nfs-storage
          persistentVolumeClaim:
            claimName: nfs-pvc
  

</TabItem>

<TabItem value="json">

{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "image": "${image_uri}",
            "resources": {
              "limits": {
                "cpu": 4,
                "memory": "12Gi",
                "nvidia.com/gpu": "{{gpus}}"
              },
              "volumeMounts": [
                {
                  "name": "nfs-storage",
                  "mountPath": "/root/.cache"
                }
              ]
            }
          }
        ],
        "restartPolicy": "Never",
        "volumes": [
          {
            "name": "nfs-storage",
            "persistentVolumeClaim": {
              "claimName": "nfs-pvc"
            }
          }
        ]
      }
    },
    "backoffLimit": 0
  }
}
  

</TabItem>

</Tabs>

이제 NFS는 우리의 작업을 실행하는 컨테이너의 `/root/.cache`에 마운트됩니다. 컨테이너가 `root`가 아닌 다른 사용자로 실행되는 경우에는 마운트 경로를 조정해야 합니다. Huggingface의 라이브러리와 W&B Artifacts는 기본적으로 `$HOME/.cache/`를 사용하므로, 다운로드는 한 번만 발생해야 합니다.

## 안정적인 확산과의 실험

새로운 시스템을 테스트하기 위해, 우리는 안정적인 확산의 추론 파라미터를 실험할 것입니다.
기본 프롬프트와 합리적인 파라미터로 간단한 안정적인 확산 추론 작업을 실행하려면, 다음을 실행할 수 있습니다:

```
wandb launch -d wandb/job_stable_diffusion_inference:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

위 코맨드는 컨테이너 이미지 `wandb/job_stable_diffusion_inference:main`을 큐에 제출할 것입니다.
에이전트가 작업을 선택하고 클러스터에서 실행되도록 스케줄링하면,
연결 상태에 따라 이미지를 가져오는 데 시간이 걸릴 수 있습니다.
[wandb.ai/launch](http://wandb.ai/launch) (혹은 wandb 서버의 경우 \<your-wandb-url\>/launch)에서 큐 페이지에서 작업 상태를 확인할 수 있습니다.

run이 완료되면, 지정한 프로젝트에 작업 아티팩트를 가지게 됩니다.
프로젝트의 작업 페이지(`<project-url>/jobs`)에서 작업 아티팩트를 찾아볼 수 있습니다. 기본 이름은
`job-wandb_job_stable_diffusion_inference`가 될 것이지만, 작업 페이지에서
작업 이름 옆의 연필 아이콘을 클릭하여 원하는 대로 변경할 수 있습니다.

이제 이 작업을 사용하여 클러스터에서 더 많은 안정적인 확산 추론을 실행할 수 있습니다!
작업 페이지에서 우리는 오른쪽 상단 모서리에 있는 **Launch** 버튼을 클릭하여
새로운 추론 작업을 구성하고 큐에 제출할 수 있습니다. 작업 구성
페이지는 원래 run의 파라미터로 미리 채워질 것이지만, 이들을
대시보드의 **Overrides** 섹션에서 값을 수정하여 원하는 대로 변경할 수 있습니다.

![안정적인 확산 추론 작업을 위한 Launch UI 이미지](/images/tutorials/minikube_gpu/sd_launch_drawer.png)