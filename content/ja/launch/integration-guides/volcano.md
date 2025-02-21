---
title: Launch multinode jobs with Volcano
menu:
  launch:
    identifier: ja-launch-integration-guides-volcano
    parent: launch-integration-guides
url: tutorials/volcano
---

このチュートリアルでは、Kubernetes 上で W&B と Volcano を使用して、マルチノードのトレーニングジョブを起動するプロセスについて説明します。

## 概要

このチュートリアルでは、W&B Launch を使用して Kubernetes 上でマルチノードジョブを実行する方法を学びます。手順は次のとおりです。

- Weights & Biases のアカウントと Kubernetes クラスターがあることを確認します。
- volcano ジョブ用の Launch キューを作成します。
- Launch エージェントを Kubernetes クラスターにデプロイします。
- 分散トレーニングジョブを作成します。
- 分散トレーニングを Launch します。

## 前提条件

始める前に、以下が必要です。

- Weights & Biases アカウント
- Kubernetes クラスター

## Launch キューを作成する

最初のステップは、Launch キューを作成することです。[wandb.ai/launch](https://wandb.ai/launch) にアクセスし、画面の右上隅にある青い **キューを作成** ボタンをクリックします。キューの作成ドロワーが画面の右側からスライドして表示されます。エンティティを選択し、名前を入力して、キューのタイプとして **Kubernetes** を選択します。

設定セクションでは、[volcano job](https://volcano.sh/en/docs/vcjob/) テンプレートを入力します。このキューから Launch された Run は、このジョブ仕様を使用して作成されるため、必要に応じてこの設定を変更してジョブをカスタマイズできます。

この設定ブロックは、Kubernetes ジョブ仕様、volcano ジョブ仕様、または Launch したいその他のカスタムリソース定義 (CRD) を受け入れることができます。[設定ブロックのマクロ]({{< relref path="/launch/set-up-launch/" lang="ja" >}}) を使用して、この仕様の内容を動的に設定できます。

このチュートリアルでは、[volcano の pytorch プラグイン](https://github.com/volcano-sh/volcano/blob/master/docs/user-guide/how_to_use_pytorch_plugin.md) を使用するマルチノード pytorch トレーニングの設定を使用します。次の構成を YAML または JSON としてコピーして貼り付けることができます。

{{< tabpane text=true >}}
{{% tab "YAML" %}}
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
{{% /tab %}}
{{% tab "JSON" %}}
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
{{% /tab %}}
{{< /tabpane >}}

ドロワーの下部にある **キューを作成** ボタンをクリックして、キューの作成を完了します。

## Volcano をインストールする

Kubernetes クラスターに Volcano をインストールするには、[公式インストールガイド](https://volcano.sh/en/docs/installation/) に従ってください。

## Launch エージェントをデプロイする

キューを作成したので、キューからジョブをプルして実行する Launch エージェントをデプロイする必要があります。最も簡単な方法は、W&B の公式 `helm-charts` リポジトリの [`launch-agent` チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使用することです。README の指示に従ってチャートを Kubernetes クラスターにインストールし、エージェントが前に作成したキューをポーリングするように構成してください。

## トレーニングジョブを作成する

Volcano の pytorch プラグインは、pytorch コードが DDP を正しく使用している限り、`MASTER_ADDR`、`RANK`、`WORLD_SIZE` など、pytorch DPP が機能するために必要な環境変数を自動的に構成します。カスタム python コードで DDP を使用する方法の詳細については、[pytorch のドキュメント](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) を参照してください。

{{% alert %}}
Volcano の pytorch プラグインは、[PyTorch Lightning `Trainer` を介したマルチノードトレーニング](https://lightning.ai/docs/pytorch/stable/common/trainer.html#num-nodes) とも互換性があります。
{{% /alert %}}

## Launch 🚀

キューとクラスターが設定されたので、分散トレーニングを Launch する時が来ました。まず、volcano の pytorch プラグインを使用して、ランダムデータで単純な多層パーセプトロンをトレーニングする [ジョブ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest) を使用します。ジョブのソースコードは[こちら](https://github.com/wandb/launch-jobs/tree/main/jobs/distributed_test)にあります。

このジョブを Launch するには、[ジョブのページ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest) にアクセスし、画面の右上隅にある **Launch** ボタンをクリックします。ジョブの Launch 元となるキューを選択するように求められます。

{{< img src="/images/launch/launching_multinode_job.png" alt="" >}}

1. ジョブの パラメータ を好きなように設定します。
2. 前に作成したキューを選択します。
3. **リソース設定** セクションで volcano ジョブを変更して、ジョブの パラメータ を変更します。たとえば、`worker` タスクの `replicas` フィールドを変更して、ワーカーの数を変更できます。
4. **Launch** 🚀 をクリックします

W&B UI からジョブの進捗状況を監視し、必要に応じてジョブを停止できます。
