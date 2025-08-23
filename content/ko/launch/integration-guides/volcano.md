---
title: Volcano 를 사용하여 멀티노드 잡 실행하기
menu:
  launch:
    identifier: ko-launch-integration-guides-volcano
    parent: launch-integration-guides
url: tutorials/volcano
---

이 튜토리얼에서는 W&B 와 Volcano 를 활용하여 Kubernetes 에서 멀티노드 트레이닝 잡을 실행하는 과정을 안내합니다.

## 개요

이 튜토리얼에서는 W&B Launch 를 사용하여 Kubernetes 에서 멀티노드 잡을 실행하는 방법을 배웁니다. 진행 순서는 다음과 같습니다.

- W&B 계정과 Kubernetes 클러스터가 있는지 확인합니다.
- volcano 잡을 위한 launch queue 를 생성합니다.
- Launch 에이전트를 kubernetes 클러스터에 배포합니다.
- 분산 트레이닝 잡을 생성합니다.
- 분산 트레이닝을 실행합니다.

## 사전 준비

시작하기 전에 아래가 준비되어 있어야 합니다:

- W&B 계정
- Kubernetes 클러스터

## launch queue 생성

첫 번째 단계는 launch queue 를 생성하는 것입니다. [wandb.ai/launch](https://wandb.ai/launch) 에 접속하여 화면 오른쪽 상단의 파란색 **Create a queue** 버튼을 클릭하세요. 오른쪽에서 queue 생성 입력창이 나타납니다. Entity 를 선택하고, 이름을 입력하고, queue 의 타입으로 **Kubernetes** 를 선택하세요.

설정 섹션에서는 [volcano job](https://volcano.sh/en/docs/vcjob/) 템플릿을 입력합니다. 이 queue 에서 실행되는 모든 run 은 해당 잡 스펙으로 생성되므로, 원하는 대로 이 설정을 수정하여 잡을 맞춤화할 수 있습니다.

이 설정 블록에는 Kubernetes 잡 스펙, volcano job 스펙 또는 실행하려는 다른 맞춤 리소스 정의(CRD)를 사용할 수 있습니다. [설정 블록에서 매크로 사용하기]({{< relref path="/launch/set-up-launch/" lang="ko" >}})를 참고하여 동적으로 스펙을 구성할 수 있습니다.

이번 튜토리얼에서는 [volcano 의 pytorch 플러그인](https://github.com/volcano-sh/volcano/blob/master/docs/user-guide/how_to_use_pytorch_plugin.md)을 활용한 멀티노드 pytorch 트레이닝 설정 예시를 사용합니다. 아래 설정을 YAML 또는 JSON 으로 복사해 사용하세요.

{{< tabpane text=true >}}
{{% tab "YAML" %}}
```yaml
# 마스터와 워커로 구성된 분산 잡 예시
kind: Job
spec:
  tasks:
    - name: master
      policies:
        - event: TaskCompleted
          action: CompleteJob
      replicas: 1
      template:
        spec:
          containers:
            - name: master
              image: ${image_uri}
              imagePullPolicy: IfNotPresent
          restartPolicy: OnFailure
    - name: worker
      replicas: 1
      template:
        spec:
          containers:
            - name: worker
              image: ${image_uri}
              workingDir: /home
              imagePullPolicy: IfNotPresent
          restartPolicy: OnFailure
  plugins:
    pytorch:
      - --master=master
      - --worker=worker
      - --port=23456
  minAvailable: 1
  schedulerName: volcano
metadata:
  name: wandb-job-${run_id}
  labels:
    wandb_entity: ${entity_name}
    wandb_project: ${project_name}
  namespace: wandb
apiVersion: batch.volcano.sh/v1alpha1
```
{{% /tab %}}
{{% tab "JSON" %}}
```json
{
  // 마스터와 워커로 구성된 분산 잡 예시
  "kind": "Job",
  "spec": {
    "tasks": [
      {
        "name": "master",
        "policies": [
          {
            "event": "TaskCompleted",
            "action": "CompleteJob"
          }
        ],
        "replicas": 1,
        "template": {
          "spec": {
            "containers": [
              {
                "name": "master",
                "image": "${image_uri}",
                "imagePullPolicy": "IfNotPresent"
              }
            ],
            "restartPolicy": "OnFailure"
          }
        }
      },
      {
        "name": "worker",
        "replicas": 1,
        "template": {
          "spec": {
            "containers": [
              {
                "name": "worker",
                "image": "${image_uri}",
                "workingDir": "/home",
                "imagePullPolicy": "IfNotPresent"
              }
            ],
            "restartPolicy": "OnFailure"
          }
        }
      }
    ],
    "plugins": {
      "pytorch": [
        "--master=master",
        "--worker=worker",
        "--port=23456"
      ]
    },
    "minAvailable": 1,
    "schedulerName": "volcano"
  },
  "metadata": {
    "name": "wandb-job-${run_id}",
    "labels": {
      "wandb_entity": "${entity_name}",
      "wandb_project": "${project_name}"
    },
    "namespace": "wandb"
  },
  "apiVersion": "batch.volcano.sh/v1alpha1"
}
```
{{% /tab %}}
{{< /tabpane >}}

입력 창 하단의 **Create queue** 버튼을 클릭하여 queue 생성을 완료하세요.

## Volcano 설치

Kubernetes 클러스터에 Volcano 를 설치하려면 [공식 설치 가이드](https://volcano.sh/en/docs/installation/)를 참고하세요.

## launch agent 배포

queue 를 생성했다면, 이제 launch agent 를 배포하여 queue 에서 잡을 받아 실행해야 합니다. 가장 쉬운 방법은 [W&B 공식 `helm-charts` 저장소의 `launch-agent` chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)를 사용하는 것입니다. README 의 안내에 따라 chart 를 Kubernetes 클러스터에 설치하시고, 에이전트가 앞서 생성한 queue 를 폴링하도록 설정하세요.

## 트레이닝 잡 생성

Volcano 의 pytorch 플러그인은 `MASTER_ADDR`, `RANK`, `WORLD_SIZE` 같은 pytorch DDP 동작에 필요한 환경 변수를 자동으로 구성해줍니다. 귀하의 pytorch 코드에서 DDP 를 올바르게 사용하고 있다면 자동으로 연동됩니다. DDP 를 활용한 파이썬 코드 예제 및 자세한 사항은 [pytorch 공식 문서](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)를 참고하세요.

{{% alert %}}
Volcano 의 pytorch 플러그인은 또한 [PyTorch Lightning `Trainer` 의 멀티노드 트레이닝](https://lightning.ai/docs/pytorch/stable/common/trainer.html#num-nodes)과도 호환됩니다.
{{% /alert %}}

## Launch

queue 와 클러스터 설정이 끝났으니, 이제 분산 트레이닝을 시작해 봅시다. 먼저 [volcano 의 pytorch 플러그인으로 임의 데이터를 사용해 간단한 다층 퍼셉트론을 트레이닝하는 잡](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest)을 사용합니다. 해당 잡의 소스 코드는 [여기](https://github.com/wandb/launch-jobs/tree/main/jobs/distributed_test)에서 확인할 수 있습니다.

이 잡을 실행하려면 [잡 페이지](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest)로 이동해, 화면 오른쪽 상단의 **Launch** 버튼을 클릭하세요. 실행할 queue 를 선택하라는 안내가 나옵니다.

{{< img src="/images/launch/launching_multinode_job.png" alt="멀티노드 job launch" >}}

1. jobs 파라미터를 원하는 대로 설정하세요.
2. 앞서 생성한 queue 를 선택하세요.
3. **Resource config** 섹션에서 volcano job 을 수정하여 잡의 파라미터를 변경할 수 있습니다. 예를 들어 `worker` task 의 `replicas` 값을 바꿔 워커 수를 조정할 수 있습니다.
4. **Launch** 를 클릭하세요.

W&B UI 에서 잡의 진행 상황을 모니터링하거나, 필요시 중단할 수 있습니다.