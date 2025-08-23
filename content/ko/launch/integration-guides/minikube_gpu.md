---
title: Minikube로 단일 노드 GPU 클러스터 실행하기
menu:
  launch:
    identifier: ko-launch-integration-guides-minikube_gpu
    parent: launch-integration-guides
url: tutorials/minikube_gpu
---

Minikube 클러스터에서 GPU 워크로드를 스케줄링하고 실행할 수 있도록 W&B Launch 를 설정하세요.

{{% alert %}}
이 튜토리얼은 여러 개의 GPU가 달린 머신에 직접 엑세스할 수 있는 사용자를 위한 가이드입니다. 클라우드 머신을 임대하는 사용자를 위한 튜토리얼이 아닙니다.

클라우드 머신에서 minikube 클러스터를 설정하려면, GPU 지원을 제공하는 클라우드 제공자의 Kubernetes 클러스터 생성을 W&B 에서 권장합니다. 예를 들어 AWS, GCP, Azure, Coreweave 등 다양한 클라우드 제공업체가 GPU 지원 Kubernetes 클러스터를 생성하는 도구를 갖추고 있습니다.

한 대의 GPU가 달린 머신에서 GPU 스케줄링을 위해 minikube 클러스터를 구성하고 싶다면, [Launch Docker queue]({{< relref path="/launch/set-up-launch/setup-launch-docker" lang="ko" >}}) 사용을 권장합니다. 물론 이 튜토리얼을 따라해보고 싶다면 해볼 수 있지만, GPU 스케줄링의 효용은 크지 않을 수 있습니다.
{{% /alert %}}

## 배경

[NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) 덕분에 Docker 환경에서 GPU 워크플로우 실행이 쉬워졌습니다. 하지만 한 가지 제약은 volume 단위의 GPU 스케줄링을 기본적으로 지원하지 않는다는 점입니다. `docker run` 코맨드에서 특정 ID의 GPU이거나, 시스템에 있는 모든 GPU를 사용할 것을 요청해야 하므로, 분산 GPU 워크로드 실행에는 한계가 있습니다. Kubernetes는 volume 요청 기반의 스케줄링을 지원하지만, 로컬 환경에서 GPU를 지원하는 Kubernetes 클러스터를 세팅하는 것은 최근까지 상당한 시간과 노력이 필요했습니다. 가장 인기 있는 단일 노드 Kubernetes 툴인 Minikube는 최근 [GPU 스케줄링을 지원](https://minikube.sigs.k8s.io/docs/tutorials/nvidia/)하기 시작했습니다. 본 튜토리얼에서는 다중 GPU 머신 위에 Minikube 클러스터를 만들고, 여기에 W&B Launch 를 사용해 여러 번의 stable diffusion inference 작업을 동시에 실행해봅니다.

## 사전 준비

시작하기 전에 다음이 필요합니다:

1. W&B 계정
2. 아래 소프트웨어가 설치 및 동작 중인 Linux 머신:
   1. Docker runtime
   2. 사용할 GPU용 드라이버
   3. Nvidia container toolkit

{{% alert %}}
튜토리얼 작성 및 테스트에는 4개의 NVIDIA Tesla T4 GPU가 장착된 `n1-standard-16` Google Cloud Compute Engine 인스턴스를 사용했습니다.
{{% /alert %}}

## launch job을 위한 큐 생성

먼저, launch job들을 위한 launch queue 를 생성합니다.

1. [wandb.ai/launch](https://wandb.ai/launch) (또는 비공개 W&B 서버를 사용하는 경우 `<your-wandb-url>/launch`)로 이동하세요.
2. 화면 오른쪽 상단의 파란색 **Create a queue** 버튼을 클릭합니다. 화면 오른쪽에서 queue 생성 창이 나타납니다.
3. 엔티티(Entity)를 선택하고, 이름을 입력하며, **Kubernetes** 를 queue 타입으로 선택하세요.
4. 오른쪽 창의 **Config** 영역에 [Kubernetes job specification](https://kubernetes.io/docs/concepts/workloads/controllers/job/)을 입력할 수 있습니다. 이 queue에서 실행되는 run은 모두 이 job spec을 활용하니, 원하는 대로 수정해 사용할 수 있습니다. 튜토리얼에서는 아래 샘플 설정을 YAML 또는 JSON으로 복사해 queue config에 붙여넣으세요:

{{< tabpane text=true >}}
{{% tab "YAML" %}}
```yaml
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
```
{{% /tab %}}
{{% tab "JSON" %}}
```json
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
```
{{% /tab %}}
{{< /tabpane >}}

queue 설정에 대한 자세한 정보는 [Set up Launch on Kubernetes]({{< relref path="/launch/set-up-launch/setup-launch-kubernetes.md" lang="ko" >}}) 및 [Advanced queue setup guide]({{< relref path="/launch/set-up-launch/setup-queue-advanced.md" lang="ko" >}})를 참고하세요.

`${image_uri}`와 `{{gpus}}`는 queue 설정에서 사용할 수 있는 두 가지 형식의 변수 템플릿입니다. `${image_uri}` 는 launch 중인 job의 image URI로, 에이전트가 자동으로 치환합니다. `{{gpus}}` 는 launch UI, CLI, SDK에서 job 제출 시 덮어쓸 수 있는 템플릿 변수입니다. 이 값들은 job spec 내에서 이미지와 GPU 자원을 조절하는 필드에 반영됩니다.

5. **Parse configuration** 버튼을 클릭해 `gpus` 템플릿 변수를 커스터마이즈하세요.
6. **Type** 을 `Integer`로, **Default**, **Min**, **Max**는 원하시는 값으로 설정하세요.
템플릿 변수 제약조건을 위반하는 run이 큐로 제출되면 거부됩니다.

{{< img src="/images/tutorials/minikube_gpu/create_queue.png" alt="Queue creation drawer" >}}

7. **Create queue**를 클릭해 큐를 생성합니다. 생성 후 새 큐의 queue 페이지로 이동됩니다.

다음 섹션에서는, 방금 만든 큐로부터 job을 가져와 실행할 수 있는 agent 설정 방법을 안내합니다.

## Docker + NVIDIA CTK 설치

이미 머신에 Docker와 Nvidia container toolkit이 설치되었다면 이 단계는 건너뛰셔도 됩니다.

Docker 설치 방법은 [Docker 공식 문서](https://docs.docker.com/engine/install/)를 참고하세요.

Docker가 설치되면, Nvidia container toolkit을 [Nvidia 공식 가이드](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)를 따라 설치하세요.

Container runtime이 GPU를 제대로 인식하는지 확인하려면 아래 명령어를 실행하세요:

```bash
docker run --gpus all ubuntu nvidia-smi
```

정상적이라면 머신에 연결된 GPU 정보를 출력하는 `nvidia-smi` 결과가 나와야 합니다. 예를 들어 다음과 같이 표시됩니다:

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

## Minikube 설치

Minikube의 GPU 지원은 `v1.32.0` 이상부터 가능합니다. 최신 설치 방법은 [Minikube 공식 문서](https://minikube.sigs.k8s.io/docs/start/)를 참고하세요. 튜토리얼에서는 아래 명령으로 최신 버전을 설치했습니다:

```yaml
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

GPU 사용을 위해 minikube 클러스터를 시작하려면 아래 명령을 입력하세요:

```yaml
minikube start --gpus all
```

명령의 출력 결과로 클러스터가 정상적으로 생성됐는지 확인할 수 있습니다.

## launch agent 시작

새 클러스터의 launch agent는 `wandb launch-agent`를 직접 실행하거나, [W&B에서 관리하는 helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)를 통해 배포할 수 있습니다.

이 튜토리얼에서는 호스트 머신에 직접 agent를 실행합니다.

{{% alert %}}
컨테이너 외부에서 agent를 실행하면, local Docker host를 사용해 클러스터에서 실행할 이미지를 빌드하는 것도 가능합니다.
{{% /alert %}}

로컬에서 agent를 실행하려면, 기본 Kubernetes API context가 Minikube 클러스터를 가리키고 있는지 확인하세요. 그런 다음 아래 명령으로 의존성을 설치하세요:

```bash
pip install "wandb[launch]"
```

agent 인증을 위해서는 `wandb login` 명령 또는 `WANDB_API_KEY` 환경변수를 등록하십시오.

agent를 시작하려면 아래 명령을 실행합니다:

```bash
wandb launch-agent -j <max-number-concurrent-jobs> -q <queue-name> -e <queue-entity>
```

터미널에 launch agent가 폴링 메시지를 출력하는 것을 볼 수 있습니다.

축하합니다. 이제 launch agent가 launch queue를 폴링하게 되었습니다. queue에 job이 추가되면, agent가 이를 감지해 Minikube 클러스터에 스케줄합니다.

## job 실행하기

이제 agent에게 job을 보내보겠습니다. 터미널에서 W&B 계정 로그인이 되어 있다면 간단한 "hello world" job을 실행할 수 있습니다:

```yaml
wandb launch -d wandb/job_hello_world:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

원하는 job이나 이미지로 테스트해도 되지만, 클러스터가 이미지를 pull할 수 있는지 확인하세요. 추가 안내는 [Minikube 공식 문서](https://minikube.sigs.k8s.io/docs/handbook/registry/)를 참고하세요. [공개 job을 테스트](https://wandb.ai/wandb/jobs/jobs?workspace=user-bcanfieldsherman)로 사용할 수도 있습니다.

## (선택) 모델 및 데이터 캐싱: NFS 활용

ML 워크로드에서는 여러 job이 동일한 데이터에 엑세스해야 할 때가 많습니다. 예컨대 데이터셋이나 모델 가중치 등 용량이 큰 파일을 반복 다운로드하지 않으려면 공용 캐시가 유용합니다. Kubernetes는 [persistent volume 및 persistent volume claim](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) 기능으로 이를 지원합니다. Persistent volume을 사용하면 Kubernetes 워크로드에서 `volumeMounts`로 직접 파일 시스템 캐시를 연결할 수 있습니다.

이 단계에서는 네트워크 파일 시스템(NFS) 서버를 구성해 모델 가중치의 공유 캐시로 사용해보겠습니다. 우선 NFS 설치 및 설정이 필요합니다. 이는 운영체제마다 다를 수 있지만, 본 튜토리얼의 VM은 Ubuntu이므로 nfs-kernel-server를 설치하고 `/srv/nfs/kubedata` 경로를 익스포트했습니다:

```bash
sudo apt-get install nfs-kernel-server
sudo mkdir -p /srv/nfs/kubedata
sudo chown nobody:nogroup /srv/nfs/kubedata
sudo sh -c 'echo "/srv/nfs/kubedata *(rw,sync,no_subtree_check,no_root_squash,no_all_squash,insecure)" >> /etc/exports'
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

호스트 파일 시스템의 export 위치와 NFS 서버의 로컬 IP 주소를 꼭 기억해 두세요. 다음 단계에서 필요합니다.

이제 이 NFS를 위한 persistent volume과 persistent volume claim을 생성해야 합니다. Persistent volume은 매우 유연하게 설정할 수 있지만, 튜토리얼에서는 단순한 구성을 사용합니다.

아래 yaml을 `nfs-persistent-volume.yaml` 파일로 복사한 후, 원하는 볼륨 용량과 클레임 요청을 설정하세요. `PersistentVolume.spec.capcity.storage`는 볼륨 최대 용량, `PersistentVolumeClaim.spec.resources.requests.storage`는 클레임 최대 용량을 의미합니다. 본 예에서는 동일한 값을 사용해도 무방합니다.

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv
spec:
  capacity:
    storage: 100Gi # 원하는 용량으로 설정하세요.
  accessModes:
    - ReadWriteMany
  nfs:
    server: <your-nfs-server-ip> # 여기에 실제 NFS서버 IP 입력.
    path: '/srv/nfs/kubedata' # 또는 본인이 설정한 경로
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
      storage: 100Gi # 원하는 용량으로 설정하세요.
  storageClassName: ''
  volumeName: nfs-pv
```

클러스터에 자원을 생성하려면 다음을 실행하세요:

```yaml
kubectl apply -f nfs-persistent-volume.yaml
```

run에서 이 캐시를 사용하려면 launch queue config에 `volumes`와 `volumeMounts`를 추가해야 합니다. launch config 수정을 위해 [wandb.ai/launch](https://wandb.ai/launch) (또는 wandb server 사용자의 경우 `<your-wandb-url>/launch`)로 이동해, 해당 큐의 **Edit config** 탭에서 원본 config을 아래와 같이 수정합니다:

{{< tabpane text=true >}}
{{% tab "YAML" %}}
```yaml
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
  backoffLimit: 0
```
{{% /tab %}}
{{% tab "JSON" %}}
```json
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
```
{{% /tab %}}
{{< /tabpane >}}

이제, 우리의 NFS가 job이 실행되는 컨테이너의 `/root/.cache`에 마운트됩니다. 컨테이너가 `root`가 아닌 다른 사용자로 실행되는 경우 마운트 경로를 바꿔야 할 수 있습니다. Huggingface 라이브러리와 W&B Artifacts는 기본적으로 `$HOME/.cache/`를 사용하므로, 데이터를 한 번만 다운로드하면 동일한 캐시를 재사용할 수 있습니다.

## stable diffusion 실험하기

시스템이 정상적으로 동작하는지 확인하고, stable diffusion의 inference 파라미터를 테스트해봅시다.
기본 프롬프트와 적당한 파라미터로 stable diffusion inference job을 손쉽게 실행하려면 다음을 입력합니다:

```
wandb launch -d wandb/job_stable_diffusion_inference:main -p <target-wandb-project> -q <your-queue-name> -e <your-queue-entity>
```

위 명령은 `wandb/job_stable_diffusion_inference:main` 이미지를 여러분의 queue로 제출합니다.
agent가 job을 감지해 클러스터에 스케줄하면, 이미지 다운로드 속도에 따라 일정 시간이 소요될 수 있습니다.
Job 진행 상황은 [wandb.ai/launch](https://wandb.ai/launch) (또는 wandb server 사용자는 \<your-wandb-url\>/launch) queue 페이지에서 확인할 수 있습니다.

run이 완료되면 지정한 프로젝트에 해당 job artifact가 남게 됩니다.
프로젝트의 job 페이지(`<project-url>/jobs`)에서 이 job artifact를 확인할 수 있습니다. 기본 이름은
`job-wandb_job_stable_diffusion_inference` 이지만, job 페이지에서 연필 아이콘을 눌러 원하는 이름으로 바꿀 수 있습니다.

이 job을 활용해 이후에도 stable diffusion inference를 반복할 수 있습니다.
job 페이지 상단 우측의 **Launch** 버튼을 클릭해, 새로운 inference job 구성을 하고 queue로 제출할 수 있습니다.
job 구성 페이지에는 원래 run의 파라미터가 미리 입력되어 있으며, **Overrides** 섹션에서 원하는 값으로 변경하면 됩니다.

{{< img src="/images/tutorials/minikube_gpu/sd_launch_drawer.png" alt="Launch UI" >}}