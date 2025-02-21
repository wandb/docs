---
title: Launch multinode jobs with Volcano
menu:
  launch:
    identifier: ja-launch-integration-guides-volcano
    parent: launch-integration-guides
url: tutorials/volcano
---

このチュートリアルでは、W&B と Kubernetes 上の Volcano を使ったマルチノードトレーニングジョブのローンチプロセスをガイドします。

## 概要

このチュートリアルでは、W&B Launch を使用して Kubernetes 上でマルチノードジョブを実行する方法を学びます。次のステップを行います。

- Weights & Biases のアカウントと Kubernetes クラスターを持っていることを確認します。
- Volcano ジョブのためのローンチキューを作成します。
- Launch エージェントを Kubernetes クラスターにデプロイします。
- 分散トレーニングジョブを作成します。
- 分散トレーニングをローンチします。

## 事前準備

始める前に、以下が必要です。

- Weights & Biases アカウント
- Kubernetes クラスター

## ローンチキューを作成する

最初のステップはローンチキューを作成することです。[wandb.ai/launch](https://wandb.ai/launch) にアクセスし、画面右上の青い **Create a queue** ボタンを押します。右側からキュー作成用の引き出しがスライドしてくるので、エンティティを選択し、名前を入力し、キューのタイプとして **Kubernetes** を選択します。

設定セクションでは、[volcano job](https://volcano.sh/en/docs/vcjob/) テンプレートを入力します。このキューからローンチされた任意の run はこのジョブ指定を使用して作成されるため、ジョブをカスタマイズするためにこの設定を必要に応じて変更できます。

この設定ブロックは、Kubernetes のジョブ仕様、volcano のジョブ仕様、または他のローンチに興味のあるカスタムリソース定義（CRD）を受け入れることができます。[設定ブロック内でのマクロの利用]({{< relref path="/launch/set-up-launch/" lang="ja" >}}) を通じて、この仕様の内容を動的に設定することができます。

このチュートリアルでは、[volcano の pytorch プラグイン](https://github.com/volcano-sh/volcano/blob/master/docs/user-guide/how_to_use_pytorch_plugin.md) を利用したマルチノードの pytorch トレーニング用の設定を使用します。以下の設定を YAML または JSON としてコピーして貼り付けることができます:

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

引き出しの下部にある **Create queue** ボタンをクリックして、キューの作成を完了させます。

## Volcano をインストールする

Kubernetes クラスターに Volcano をインストールするには、[公式のインストールガイド](https://volcano.sh/en/docs/installation/)に従ってください。

## launch agent をデプロイする

キューを作成したので、キューからジョブをプルして実行するための launch agent をデプロイする必要があります。これを行う最も簡単な方法は、W&B の公式 `helm-charts` リポジトリから [`launch-agent` chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使用することです。README の指示に従ってチャートを Kubernetes クラスターにインストールし、作成したキューをポーリングするようにエージェントを設定してください。

## トレーニングジョブを作成する

Volcano の pytorch プラグインは、pytorch の DPP が動作するために必要な環境変数（例: `MASTER_ADDR`、`RANK`、`WORLD_SIZE`）を自動的に設定します。ただし、pytorch のコードが正しく DDP を使用する必要があります。カスタム Python コードで DDP を使用する方法については、[pytorch のドキュメント](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)を参照してください。

{{% alert %}}
Volcano の pytorch プラグインは、[PyTorch Lightning の `Trainer` によるマルチノードトレーニング](https://lightning.ai/docs/pytorch/stable/common/trainer.html#num-nodes)にも対応しています。
{{% /alert %}}

## ローンチ 🚀

キューとクラスターが設定されたので、分散トレーニングをローンチする準備が整いました。まず、Volcano の pytorch プラグインを使用してランダムデータ上で単純なマルチレイヤーパーセプトロンをトレーニングする [ジョブ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest) を使用します。このジョブのソースコードは [こちら](https://github.com/wandb/launch-jobs/tree/main/jobs/distributed_test) で見つけることができます。

このジョブをローンチするには、[ジョブのページ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest)にアクセスし、画面の右上にある **Launch** ボタンをクリックします。ジョブをローンチするキューを選択するように促されます。

{{< img src="/images/launch/launching_multinode_job.png" alt="" >}}

1. ジョブのパラメータを任意で設定します。
2. 作成したキューを選択します。
3. **Resource config** セクションで Volcano ジョブを変更して、ジョブのパラメータを変更します。例えば、`worker` タスクの `replicas` フィールドを変更してワーカー数を変更できます。
4. **Launch** をクリックしてローンチしてください 🚀

進捗を監視し、必要に応じてジョブを W&B UI から停止することができます。