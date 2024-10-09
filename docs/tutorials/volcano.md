---
title: Launch multinode jobs with Volcano
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

이 튜토리얼은 Kubernetes에서 W&B와 Volcano를 사용하여 다중 노드 트레이닝 작업을 시작하는 과정을 안내합니다.

## 개요

이 튜토리얼에서는 W&B Launch를 사용하여 Kubernetes에서 다중 노드 작업을 실행하는 방법을 배우게 됩니다. 우리가 따를 단계는 다음과 같습니다:

- Weights & Biases 계정과 Kubernetes 클러스터가 있는지 확인합니다.
- Volcano 작업을 위한 launch 큐를 만듭니다.
- Kubernetes 클러스터에 Launch 에이전트를 배포합니다.
- 분산 트레이닝 작업을 생성합니다.
- 분산 트레이닝을 시작합니다.

## 사전 준비사항

시작하기 전에 필요한 것들:

- Weights & Biases 계정
- Kubernetes 클러스터

## Launch 큐 생성하기

첫 번째 단계는 Launch 큐를 생성하는 것입니다. [wandb.ai/launch](https://wandb.ai/launch)로 이동하여 화면의 오른쪽 상단 코너에서 파란색 **Create a queue** 버튼을 누르세요. 큐 생성 서랍이 화면 오른쪽에서 슬라이드로 나옵니다. Entity를 선택하고 이름을 입력한 다음, 큐의 타입으로 **Kubernetes**를 선택하세요.

설정 섹션에서 [volcano job](https://volcano.sh/en/docs/vcjob/) 템플릿을 입력할 것입니다. 이 큐에서 실행되는 모든 runs는 이 작업 명세를 사용하여 생성되므로, 필요에 따라 이 설정을 수정하여 작업을 맞춤화할 수 있습니다.

이 설정 블록은 Kubernetes 작업 명세, volcano 작업 명세, 또는 시작하고자 하는 다른 맞춤 리소스 정의(CRD)를 사용할 수 있습니다. [설정 블록에서 매크로 사용하기](../guides/launch/setup-launch.md)를 통해 이 스펙의 내용을 동적으로 설정할 수 있습니다.

이 튜토리얼에서는 [volcano의 pytorch 플러그인](https://github.com/volcano-sh/volcano/blob/master/docs/user-guide/how_to_use_pytorch_plugin.md)을 사용하는 다중 노드 pytorch 트레이닝을 위한 설정을 사용할 것입니다. 아래의 설정을 YAML 또는 JSON으로 복사하여 붙여넣을 수 있습니다:

<Tabs
defaultValue="yaml"
values={[
{ label: "YAML", value: "yaml", },
{ label: "JSON", value: "json", },
]}>

<TabItem value="yaml">

```yaml
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

</TabItem>

<TabItem value="json">

```json
{
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

</TabItem>

</Tabs>

서랍 하단의 **Create queue** 버튼을 클릭하여 큐 생성 작업을 완료하세요.

## Volcano 설치하기

Kubernetes 클러스터에 Volcano를 설치하려면 [공식 설치 가이드](https://volcano.sh/en/docs/installation/)를 따르시면 됩니다.

## Launch 에이전트 배포하기

이제 큐를 생성했으니, 큐에서 작업을 가져와 실행하기 위한 Launch 에이전트를 배포해야 합니다. 가장 쉬운 방법은 W&B의 공식 `helm-charts` 저장소에서 제공하는 [`launch-agent` 차트](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)를 사용하는 것입니다. README에 있는 지침을 따라 Kubernetes 클러스터에 차트를 설치하고, 이전에 생성한 큐를 폴링할 수 있도록 에이전트를 설정하세요.

## 트레이닝 작업 생성하기

Volcano의 pytorch 플러그인은 pytorch ddp가 작동하는 데 필요한 환경 변수를 자동으로 설정합니다, 예를 들어 `MASTER_ADDR`, `RANK`, `WORLD_SIZE` 등입니다. pytorch 코드가 DDP를 올바르게 사용하면, 나머지는 **문제없이 작동**해야 합니다. 사용자 정의 python 코드에서 DDP를 사용하는 방법에 대한 자세한 내용은 [pytorch의 문서](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)를 참조하세요.

:::tip
Volcano의 pytorch 플러그인은 [PyTorch Lightning `Trainer`를 통한 다중 노드 트레이닝](https://lightning.ai/docs/pytorch/stable/common/trainer.html#num-nodes)과도 호환됩니다.
:::

## 시작하기 🚀

이제 큐와 클러스터가 설정되었으므로, 분산 트레이닝을 시작할 때입니다! 먼저 [작업](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest)을 사용하여 Volcano의 pytorch 플러그인을 사용하여 랜덤 데이터로 간단한 다층 퍼셉트론을 트레이닝합니다. 작업의 소스 코드는 [여기](https://github.com/wandb/launch-jobs/tree/main/jobs/distributed_test)에서 찾을 수 있습니다.

이 작업을 시작하려면 [작업 페이지](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest)로 이동하여 화면 오른쪽 상단의 **Launch** 버튼을 클릭하세요. 작업을 시작할 큐를 선택하라는 메시지가 나타납니다.

1. 작업의 파라미터를 원하는 대로 설정하세요,
2. 이전에 생성한 큐를 선택하세요.
3. 작업의 파라미터를 수정하려면 **Resource config** 섹션에서 volcano 작업을 수정하세요. 예를 들어, `worker` 작업의 `replicas` 필드를 변경하여 작업자 수를 변경할 수 있습니다.
4. **Launch** 클릭 🚀

W&B UI에서 작업의 진행 상황을 모니터링하거나 필요에 따라 작업을 중단할 수 있습니다.