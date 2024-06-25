import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Volcanoを使ったマルチノードジョブの起動

このチュートリアルでは、Kubernetes上でW&BとVolcanoを使ってマルチノードトレーニングジョブを起動する手順を紹介します。

## 概要

このチュートリアルでは、W&B Launchを使ってKubernetes上でマルチノードジョブを起動する方法を学びます。以下のステップに従います：

- Weights & BiasesアカウントとKubernetesクラスターを用意する
- Volcanoジョブ用のローンチキューを作成する
- KubernetesクラスターにLaunchエージェントをデプロイする
- 分散トレーニングジョブを作成する
- 分散トレーニングを開始する

## 前提条件

開始する前に、以下が必要です：

- Weights & Biasesアカウント
- Kubernetesクラスター

## ローンチキューを作成

最初のステップはローンチキューを作成することです。[wandb.ai/launch](https://wandb.ai/launch)にアクセスし、画面の右上にある青い**Create a queue**ボタンをクリックします。右側からキュー作成ドロワーが表示されます。エンティティを選択し、名前を入力し、キューのタイプとして**Kubernetes**を選択します。

設定セクションには、[volcano job](https://volcano.sh/en/docs/vcjob/)テンプレートを入力します。ここから起動する任意のrunsは、このジョブ仕様を使用して作成されるため、この設定をカスタマイズしてジョブを調整することができます。

この設定ブロックは、Kubernetesジョブ仕様、Volcanoジョブ仕様、または他の任意のカスタムリソース定義（CRD）を受け入れることができます。[設定ブロックでのマクロの使用](../guides/launch/setup-launch.md)により、このスペックの内容を動的に設定できます。

このチュートリアルでは、[volcanoのpytorchプラグイン](https://github.com/volcano-sh/volcano/blob/master/docs/user-guide/how_to_use_pytorch_plugin.md)を使用したマルチノードpytorchトレーニングの設定を使用します。以下の設定をYAMLまたはJSONとしてコピーして貼り付けることができます：

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

ドロワーの下部にある**Create queue**ボタンをクリックして、キューの作成を完了します。

## Volcanoのインストール

KubernetesクラスターにVolcanoをインストールするには、[公式インストールガイド](https://volcano.sh/en/docs/installation/)に従ってください。

## Launchエージェントのデプロイ

キューを作成したので、次は作成したキューからジョブを取得して実行するLaunchエージェントをデプロイする必要があります。最も簡単な方法は、W&Bの公式[`launch-agent`チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)を使うことです。READMEの指示に従って、このチャートをKubernetesクラスターにインストールし、先ほど作成したキューをポーリングするようにエージェントを設定してください。

## トレーニングジョブの作成

Volcanoのpytorchプラグインは、pytorch ddpが動作するために必要な環境変数（例：`MASTER_ADDR`、`RANK`、`WORLD_SIZE`など）を自動的に設定します。pytorchコードが正しくDDPを使用している限り、他の操作は**そのまま動作**します。DDPの使用方法については、[pytorchのドキュメント](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)を参照してください。

:::tip
Volcanoのpytorchプラグインは、[PyTorch Lightningの `Trainer` を使ったマルチノードトレーニング](https://lightning.ai/docs/pytorch/stable/common/trainer.html#num-nodes)とも互換性があります。
:::

## 起動 🚀

キューとクラスターの設定が完了したので、いよいよ分散トレーニングを開始します！最初に使用するのは、volcanoのpytorchプラグインを使ってランダムデータ上でシンプルな多層パーセプトロンをトレーニングする[ジョブ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest)です。このジョブのソースコードは[こちら](https://github.com/wandb/launch-jobs/tree/main/jobs/distributed_test)から入手できます。

このジョブを起動するには、[ジョブのページ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest)にアクセスし、画面の右上の**Launch**ボタンをクリックします。ジョブを起動するキューを選択するように求められます。

![](/images/launch/launching_multinode_job.png)

1. 任意のジョブパラメータを設定します。
2. 先ほど作成したキューを選択します。
3. **Resource config**セクションでvolcanoジョブを修正し、ジョブのパラメータを変更します。例えば、`worker`タスクの`replicas`フィールドを変更してワーカーの数を変更できます。
4. **Launch**をクリックします 🚀

W&B UIからジョブの進行状況を監視し、必要に応じてジョブを停止することができます。