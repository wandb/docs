---
title: Volcano でマルチノードジョブをローンチする
menu:
  launch:
    identifier: ja-launch-integration-guides-volcano
    parent: launch-integration-guides
url: tutorials/volcano
---

このチュートリアルでは、Kubernetes上でW&BとVolcanoを使用してマルチノードトレーニングのジョブをローンチするプロセスを説明します。

## 概要

このチュートリアルでは、W&B Launchを使用してKubernetes上でマルチノードジョブを実行する方法を学びます。私たちが従うステップは以下の通りです：

- Weights & BiasesのアカウントとKubernetesクラスターを確認する。
- Volcanoジョブ用のローンチキューを作成する。
- KubernetesクラスターにLaunchエージェントをデプロイする。
- 分散トレーニングジョブを作成する。
- 分散トレーニングをローンチする。

## 必要条件

開始する前に必要なもの：

- Weights & Biasesアカウント
- Kubernetesクラスター

## ローンチキューを作成する

最初のステップはローンチキューを作成することです。[wandb.ai/launch](https://wandb.ai/launch)にアクセスし、画面の右上隅にある青い**Create a queue**ボタンを押します。右側からキュー作成ドロワーがスライドアウトします。エンティティを選択し、名前を入力し、キューのタイプとして**Kubernetes**を選択します。

設定セクションで、[volcanoのジョブ](https://volcano.sh/en/docs/vcjob/)のテンプレートを入力します。このキューからローンチされたすべてのrunはこのジョブ仕様を使用して作成されるため、ジョブをカスタマイズしたい場合はこの設定を変更できます。

この設定ブロックには、Kubernetesジョブ仕様、volcanoジョブ仕様、または他のカスタムリソース定義（CRD）をローンチするために使用することができます。[設定ブロック内のマクロ]({{< relref path="/launch/set-up-launch/" lang="ja" >}})を利用して、この仕様の内容を動的に設定することができます。

このチュートリアルでは、[volcanoのpytorchプラグイン](https://github.com/volcano-sh/volcano/blob/master/docs/user-guide/how_to_use_pytorch_plugin.md)を利用したマルチノードpytorchトレーニングの設定を使用します。以下の設定をYAMLまたはJSONとしてコピーして貼り付けることができます：

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

ドロワーの下部にある**Create queue**ボタンをクリックしてキューの作成を完了します。

## Volcanoをインストールする

KubernetesクラスターにVolcanoをインストールするには、[公式インストールガイド](https://volcano.sh/en/docs/installation/)に従ってください。

## ローンチエージェントをデプロイする

キューを作成した後は、キューからジョブを引き出して実行するためにローンチエージェントをデプロイする必要があります。これを行う最も簡単な方法は、W&Bの公式`helm-charts`リポジトリから[`launch-agent`チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)を使用することです。READMEに記載された指示に従って、Kubernetesクラスターにチャートをインストールし、エージェントが先ほど作成したキューをポーリングするように設定してください。

## トレーニングジョブを作成する

Volcanoのpytorchプラグインは、pytorch DPPが機能するために必要な環境変数（`MASTER_ADDR`、`RANK`、`WORLD_SIZE`など）を自動で設定します。ただし、pytorchコードがDDPを正しく使用している場合に限ります。カスタムのPythonコードでDDPを使用する方法の詳細については、[pytorchのドキュメント](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)を参照してください。

{{% alert %}}
Volcanoのpytorchプラグインは、[PyTorch Lightning `Trainer`を使用したマルチノードトレーニング](https://lightning.ai/docs/pytorch/stable/common/trainer.html#num-nodes)とも互換性があります。
{{% /alert %}}

## ローンチ 🚀

キューとクラスターのセットアップが完了したので、分散トレーニングを開始する時がきました。最初に、Volcanoのpytorchプラグインを使用してランダムデータ上でシンプルなマルチレイヤパーセプトロンをトレーニングする[a job](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest)を使用します。このジョブのソースコードは[こちら](https://github.com/wandb/launch-jobs/tree/main/jobs/distributed_test)で見つけることができます。

このジョブをローンチするには、[ジョブのページ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest)にアクセスし、画面の右上にある**Launch**ボタンをクリックします。ジョブをローンチするキューを選択するように促されます。

{{< img src="/images/launch/launching_multinode_job.png" alt="" >}}

1. ジョブのパラメータを好きなように設定し、
2. 先ほど作成したキューを選択します。
3. **Resource config**セクションでVolcanoジョブを変更してジョブのパラメータを変更します。例えば、`worker`タスクの`replicas`フィールドを変更することによってワーカーの数を変更できます。
4. **Launch**をクリック 🚀

W&B UIからジョブの進捗をモニターし、必要に応じてジョブを停止できます。