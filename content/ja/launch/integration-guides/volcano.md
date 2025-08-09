---
title: Volcano でマルチノードジョブをローンンチする
menu:
  launch:
    identifier: ja-launch-integration-guides-volcano
    parent: launch-integration-guides
url: tutorials/volcano
---

このチュートリアルでは、W&B と Volcano を使って Kubernetes 上でマルチノードのトレーニングジョブをローンンチする方法を案内します。

## 概要

このチュートリアルでは、W&B Launch を利用して Kubernetes 上でマルチノードジョブを実行する手順を学びます。流れは以下の通りです。

- W&B アカウントと Kubernetes クラスターを用意する
- volcano ジョブ用の launch queue を作成する
- Launch エージェントを Kubernetes クラスターにデプロイする
- 分散トレーニングジョブを作成する
- 分散トレーニングをローンンチする

## 必要条件

始める前に、以下が必要です。

- W&B アカウント
- Kubernetes クラスター

## launch queue の作成

まず初めに launch queue を作成します。[wandb.ai/launch](https://wandb.ai/launch) へアクセスし、画面右上の青い **Create a queue** ボタンをクリックします。右側から queue 作成用のドロワーがスライドして開くので、Entity を選び、名前を入力し、queue のタイプとして **Kubernetes** を選択してください。

設定セクションでは、[volcano job](https://volcano.sh/en/docs/vcjob/) テンプレートを入力します。この queue からローンンチされるすべての run はこのジョブ定義で作成されるため、必要に応じて設定内容を調整してカスタマイズできます。

この設定ブロックには Kubernetes job スペック、volcano job スペック、または他の任意のカスタムリソース定義（CRD）を指定できます。[設定ブロック内でマクロを利用することで]({{< relref path="/launch/set-up-launch/" lang="ja" >}})、この spec の内容を動的に制御できます。

本チュートリアルでは、[volcano の pytorch プラグイン](https://github.com/volcano-sh/volcano/blob/master/docs/user-guide/how_to_use_pytorch_plugin.md)を利用したマルチノード pytorch トレーニング用の設定例を使います。以下の設定を YAML または JSON 形式でコピー&ペーストしてください。

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

ドロワー下部の **Create queue** ボタンを押して queue の作成を完了します。

## Volcano のインストール

Kubernetes クラスターに Volcano をインストールするには、[公式インストールガイド](https://volcano.sh/en/docs/installation/)の手順に従ってください。

## Launch エージェントのデプロイ

queue を作成したら、その queue からジョブを取得・実行するための launch エージェントをデプロイする必要があります。最も簡単な方法は、[W&B 公式の `helm-charts` リポジトリ内の `launch-agent` チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を利用することです。README の手順に従って Kubernetes クラスターにチャートをインストールし、エージェントが先ほど作成した queue をポーリングするように設定しましょう。

## トレーニングジョブの作成

Volcano の pytorch プラグインでは、pytorch DDP 用に必要な環境変数（`MASTER_ADDR`, `RANK`, `WORLD_SIZE` など）が自動的に設定されます。あなたの pytorch コードで DDP を正しく利用していれば問題ありません。カスタム python コードでの DDP 利用方法については、[pytorch のドキュメント](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)を参照してください。

{{% alert %}}
Volcano の pytorch プラグインは、[PyTorch Lightning の `Trainer` を使ったマルチノードトレーニング](https://lightning.ai/docs/pytorch/stable/common/trainer.html#num-nodes)にも対応しています。
{{% /alert %}}

## Launch

queue とクラスターのセットアップが完了したので、いよいよ分散トレーニングをローンンチします。まずは、[このジョブ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest)（volcano の pytorch プラグインを使ってランダムデータで多層パーセプトロンをトレーニングします）を使いましょう。ジョブのソースコードは[こちら](https://github.com/wandb/launch-jobs/tree/main/jobs/distributed_test)で確認できます。

ジョブをローンンチするには、[ジョブのページ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest) にアクセスし、画面右上の **Launch** ボタンをクリックします。どの queue からローンンチするか選択を求められます。

{{< img src="/images/launch/launching_multinode_job.png" alt="Multi-node job launch" >}}

1. ジョブのパラメータを好きなように設定します。
2. 先ほど作成した queue を選択します。
3. **Resource config** セクション内の volcano ジョブを編集して、ジョブのパラメータを変更します。たとえば `worker` タスクの `replicas` を変更するとワーカー数を調整できます。
4. **Launch** をクリックします。

W&B の UI から進捗の監視や、必要に応じてジョブの停止も可能です。