import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Volcanoでマルチノードジョブを起動する

このチュートリアルでは、Kubernetes上でW&BとVolcanoを使用してマルチノードのトレーニングジョブを起動する手順を案内します。

## 概要

このチュートリアルでは、W&B Launchを使用してKubernetes上でマルチノードのジョブを実行する方法を学びます。以下のステップを進めます：

- Weights & BiasesのアカウントとKubernetesクラスターを確認する。
- VolcanoジョブのためのLaunchキューを作成する。
- LaunchエージェントをKubernetesクラスターにデプロイする。
- 分散トレーニングジョブを作成する。
- 分散トレーニングを起動する。

## 前提条件

開始する前に、以下が必要です：

- Weights & Biasesのアカウント
- Kubernetesクラスター

## Launchキューを作成する

最初のステップはLaunchキューを作成することです。[wandb.ai/launch](https://wandb.ai/launch)にアクセスし、画面の右上にある青い**Create a queue**ボタンをクリックします。画面の右側からキュー作成ドロワーがスライドアウトします。エンティティを選択し、名前を入力し、キューのタイプとして**Kubernetes**を選択します。

設定セクションでは、[volcano job](https://volcano.sh/en/docs/vcjob/)テンプレートを入力します。このキューから起動する任意のrunはこのジョブ仕様を使用して作成されるため、ジョブをカスタマイズするためにこの設定を必要に応じて変更できます。

この設定ブロックは、Kubernetesジョブ仕様、volcanoジョブ仕様、または他のカスタムリソース定義（CRD）を受け入れることができます。設定ブロック内では[マクロを使用](../guides/launch/setup-launch.md)して、この仕様の内容を動的に設定することができます。

このチュートリアルでは、[volcanoのpytorchプラグイン](https://github.com/volcano-sh/volcano/blob/master/docs/user-guide/how_to_use_pytorch_plugin.md)を使用したマルチノードpytorchトレーニングの設定を使用します。以下の設定をYAMLまたはJSONとしてコピー＆ペーストできます：

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

## Volcanoをインストールする

KubernetesクラスターにVolcanoをインストールするには、[公式インストールガイド](https://volcano.sh/en/docs/installation/)に従ってください。

## Launchエージェントをデプロイする

キューを作成したので、次にジョブをキューから取得して実行するLaunchエージェントをデプロイする必要があります。最も簡単な方法は、W&Bの公式`helm-charts`リポジトリにある[`launch-agent`チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)を使用することです。READMEの指示に従って、Kubernetesクラスターにチャートをインストールし、エージェントが先ほど作成したキューをポーリングするように設定してください。

## トレーニングジョブを作成する

Volcanoのpytorchプラグインは、pytorch ddpが動作するために必要な環境変数、例えば`MASTER_ADDR`、`RANK`、`WORLD_SIZE`などを自動的に設定します。pytorchのコードをDDPに適切に対応させる限り、他のすべては**ただ動作する**でしょう。詳細については、[pytorchのドキュメント](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)を参照してください。

:::tip
Volcanoのpytorchプラグインは、[PyTorch Lightning `Trainer`によるマルチノードトレーニング](https://lightning.ai/docs/pytorch/stable/common/trainer.html#num-nodes)にも対応しています。
:::

## 起動 🚀

キューとクラスターが設定されたので、分散トレーニングを起動する時が来ました！最初に、volcanoのpytorchプラグインを使用してランダムデータに対してシンプルな多層パーセプトロンをトレーニングする[ジョブ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest)を使用します。このジョブのソースコードは[こちら](https://github.com/wandb/launch-jobs/tree/main/jobs/distributed_test)で確認できます。

このジョブを起動するには、[ジョブのページ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest)に移動し、画面右上の**Launch**ボタンをクリックします。ジョブを起動するキューを選択するように求められます。

![](/images/launch/launching_multinode_job.png)

1. ジョブのパラメーターを任意に設定します。
2. 先ほど作成したキューを選択します。
3. **Resource config**セクションでvolcanoジョブを変更し、ジョブのパラメーターを変更します。例えば、`worker`タスクの`replicas`フィールドを変更してワーカーの数を変更できます。
4. **Launch**をクリックします 🚀

W&BのUIからジョブの進行状況を監視し、必要に応じてジョブを停止することができます。