import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Volcanoを使ったマルチノードジョブの起動

このチュートリアルでは、Kubernetes上でW&BとVolcanoを使用してマルチノードトレーニングジョブを起動する方法をガイドします。

## 概要

このチュートリアルでは、W&B Launchを使用してKubernetes上でマルチノードジョブを実行する方法を学びます。以下の手順に従います：

- Weights & BiasesアカウントとKubernetesクラスターが必要です。
- Volcanoジョブのためのキューを作成します。
- KubernetesクラスターにLaunchエージェントをデプロイします。
- 分散トレーニングジョブを作成します。
- 分散トレーニングを開始します。

## 前提条件

始める前に必要なもの:

- Weights & Biasesアカウント
- Kubernetesクラスター

## キューの作成

最初のステップはキューを作成することです。[wandb.ai/launch](https://wandb.ai/launch)にアクセスし、画面の右上にある青い**Create a queue**ボタンをクリックします。右側からキュー作成のドロワーが表示されます。エンティティを選択し、名前を入力し、キューのタイプとして**Kubernetes**を選択します。

設定セクションでは、[volcano job](https://volcano.sh/en/docs/vcjob/)テンプレートを入力します。このキューから起動されるすべてのrunはこのジョブ仕様を使用して作成されるため、必要に応じてこの設定を修正してジョブをカスタマイズできます。

この設定ブロックは、Kubernetesジョブ仕様、volcanoジョブ仕様、または起動したい任意の他のカスタムリソース定義（CRD）を受け入れることができます。設定ブロックの[マクロを使用](../guides/launch/setup-launch.md)して、このスペックの内容を動的に設定できます。

このチュートリアルでは、[volcanoのpytorchプラグイン](https://github.com/volcano-sh/volcano/blob/master/docs/user-guide/how_to_use_pytorch_plugin.md)を利用したマルチノードpytorchトレーニングの設定を使用します。以下の設定をYAMLまたはJSONとしてコピー＆ペーストできます：

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

ドロワーの下部にある**Create queue**ボタンをクリックしてキューの作成を完了します。

## Volcanoのインストール

KubernetesクラスターにVolcanoをインストールするには、[公式インストールガイド](https://volcano.sh/en/docs/installation/)に従ってください。

## Launchエージェントのデプロイ

キューを作成した後、そのキューからジョブをプルして実行するためにLaunchエージェントをデプロイする必要があります。最も簡単な方法は、[`launch-agent`チャートをW&Bの公式`helm-charts`リポジトリからインストールする](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)ことです。READMEの指示に従って、このチャートをKubernetesクラスターにインストールし、先に作成したキューをポーリングするようにエージェントを設定してください。

## トレーニングジョブの作成

Volcanoのpytorchプラグインは、pytorch ddpが機能するために必要な環境変数（例：`MASTER_ADDR`, `RANK`, `WORLD_SIZE`など）を自動的に設定します。あなたのpytorchコードがDDPを正しく使用するように書かれていれば、他の部分は**うまく動作するはずです**。DDPをカスタムのpythonコードで使用する方法については、[pytorchのドキュメント](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)を参照してください。

:::tip
Volcanoのpytorchプラグインは、[PyTorch Lightningの`Trainer`によるマルチノードトレーニング](https://lightning.ai/docs/pytorch/stable/common/trainer.html#num-nodes)にも対応しています。
:::

## 起動 🚀

キューとクラスターが設定されたので、分散トレーニングを開始する時が来ました！最初に、[こちらのジョブ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest)を使用して、volcanoのpytorchプラグインを使ってランダムデータに対してシンプルな多層パーセプトロンをトレーニングします。このジョブのソースコードは[こちら](https://github.com/wandb/launch-jobs/tree/main/jobs/distributed_test)で確認できます。

このジョブを起動するには、[ジョブのページ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest)にアクセスし、画面の右上にある**Launch**ボタンをクリックします。ジョブを起動するキューを選択するように求められます。

![](/images/launch/launching_multinode_job.png)

1. ジョブのパラメーターを好きなように設定します。
2. 先に作成したキューを選択します。
3. **Resource config**セクションでvolcanoジョブを修正して、ジョブのパラメーターを変更します。例えば、`worker`タスクの`replicas`フィールドを変更してワーカーの数を変えることができます。
4. **Launch**をクリック 🚀

W&BのUIから、進行状況を監視したり、必要に応じてジョブを停止することができます。