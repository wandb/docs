---
title: Volcano でマルチノード ジョブを Launch する
menu:
  launch:
    identifier: ja-launch-integration-guides-volcano
    parent: launch-integration-guides
url: tutorials/volcano
---

このチュートリアルでは、Kubernetes 上で W&B と Volcano を使ってマルチノードのトレーニング ジョブを起動する手順を説明します。

## 概要

このチュートリアルでは、W&B Launch を使って Kubernetes 上でマルチノード ジョブを実行する方法を学びます。進める手順は次のとおりです。

- W&B アカウントと Kubernetes クラスターを用意していることを確認する。
- Volcano ジョブ用の Launch queue を作成する。
- Kubernetes クラスターに Launch agent をデプロイする。
- 分散トレーニング ジョブを作成する。
- 分散トレーニングを Launch する。

## 前提条件

開始する前に、以下が必要です。

- W&B アカウント
- Kubernetes クラスター

## Launch queue を作成する

最初のステップは Launch queue を作成することです。[wandb.ai/launch](https://wandb.ai/launch) にアクセスし、画面右上の青い **Create a queue** ボタンをクリックします。画面右側からキュー作成用のドロワーが開きます。Entity を選び、名前を入力し、タイプとして **Kubernetes** を選択します。

設定セクションには、[volcano job](https://volcano.sh/en/docs/vcjob/) のテンプレートを入力します。この queue から Launch された Runs は、このジョブ仕様に基づいて作成されます。必要に応じてこの設定を変更して、ジョブをカスタマイズできます。

この設定ブロックには、Kubernetes の Job specification、Volcano の Job specification、または起動したい任意の CRD を記述できます。仕様の内容を動的に設定するために [設定ブロックのマクロ]({{< relref path="/launch/set-up-launch/" lang="ja" >}}) を利用できます。

このチュートリアルでは、[Volcano の PyTorch プラグイン](https://github.com/volcano-sh/volcano/blob/master/docs/user-guide/how_to_use_pytorch_plugin.md) を使ったマルチノード PyTorch トレーニング用の設定を使用します。以下の設定を YAML または JSON としてコピー＆ペーストできます。

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

ドロワーの一番下にある **Create queue** をクリックして、queue の作成を完了します。

## Volcano をインストールする

Kubernetes クラスターに Volcano をインストールするには、[公式インストール ガイド](https://volcano.sh/en/docs/installation/) に従ってください。

## Launch agent をデプロイする

queue を作成したら、queue からジョブを取得して実行するために Launch agent をデプロイする必要があります。最も簡単な方法は、W&B 公式の `helm-charts` リポジトリにある [`launch-agent` チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使うことです。README の手順に従ってチャートを Kubernetes クラスターにインストールし、先ほど作成した queue をポーリングするように agent を設定してください。

## トレーニング ジョブを作成する

Volcano の PyTorch プラグインは、あなたの PyTorch コードが正しく DDP を使っていれば、`MASTER_ADDR`、`RANK`、`WORLD_SIZE` など PyTorch DDP に必要な環境変数を自動的に設定します。カスタムの Python コードで DDP を使用する方法の詳細は、[PyTorch のドキュメント](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)を参照してください。

{{% alert %}}
Volcano の PyTorch プラグインは、[PyTorch Lightning の `Trainer` によるマルチノード トレーニング](https://lightning.ai/docs/pytorch/stable/common/trainer.html#num-nodes) とも互換性があります。
{{% /alert %}}

## Launch

queue とクラスターの準備ができたので、分散トレーニングを Launch してみましょう。まずは、Volcano の PyTorch プラグインを使い、ランダムなデータに対して単純な多層パーセプトロンを学習させる [ジョブ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest) を使います。ジョブのソースコードは [こちら](https://github.com/wandb/launch-jobs/tree/main/jobs/distributed_test) にあります。

このジョブを Launch するには、[ジョブのページ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest) にアクセスし、画面右上の **Launch** ボタンをクリックします。どの queue からジョブを Launch するかの選択を求められます。

{{< img src="/images/launch/launching_multinode_job.png" alt="マルチノード ジョブの Launch" >}}

1. ジョブのパラメータを任意に設定します。
2. 先ほど作成した queue を選択します。
3. **Resource config** セクションの Volcano ジョブを編集して、ジョブのパラメータを変更します。例えば、`worker` タスクの `replicas` フィールドを変更すると worker の数を変えられます。
4. **Launch** をクリックします。

進捗の監視や、必要に応じたジョブの停止は、W&B の UI から行えます。