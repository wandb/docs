---
title: Volcano を使ってマルチノードジョブをローンチする
menu:
  launch:
    identifier: volcano
    parent: launch-integration-guides
url: tutorials/volcano
---

このチュートリアルでは、W&B と Volcano を使って Kubernetes 上でマルチノードのトレーニングジョブをローンチする手順を紹介します。

## 概要

このチュートリアルでは、W&B Launch を利用して Kubernetes 上でマルチノードジョブを実行する方法を学びます。手順は以下の通りです。

- W&B アカウントと Kubernetes クラスターを用意する
- Volcano ジョブ用の launch キューを作成する
- Launch エージェントを kubernetes クラスターにデプロイする
- 分散トレーニングジョブを作成する
- 作成した分散トレーニングをローンチする

## 前提条件

始める前に、以下をご用意ください。

- W&B アカウント
- Kubernetes クラスター

## launch キューの作成

最初のステップは launch キューの作成です。[wandb.ai/launch](https://wandb.ai/launch) にアクセスし、右上の青い **Create a queue** ボタンをクリックしてください。画面右側からキュー作成用のドロワーがスライドして開きます。Entity を選択し、キュー名を入力し、キューのタイプとして **Kubernetes** を選択します。

設定セクションでは、[volcano job](https://volcano.sh/en/docs/vcjob/) のテンプレートを入力します。このキューから launch される Run は、このジョブの設定で作成されます。必要に応じて設定を修正してジョブをカスタマイズできます。

この設定ブロックには、Kubernetes のジョブ仕様や、volcano ジョブ仕様、その他カスタムリソース定義（CRD）など、ローンチしたいものを記載できます。設定ブロック内で [マクロ]({{< relref "/launch/set-up-launch/" >}}) を使うと、この spec の内容を動的に設定できます。

このチュートリアルでは、[volcano の pytorch プラグイン](https://github.com/volcano-sh/volcano/blob/master/docs/user-guide/how_to_use_pytorch_plugin.md)を利用したマルチノード pytorch トレーニング用の設定例を使います。下記の YAML または JSON のいずれかをコピー＆ペーストしてください。

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

ドロワーの一番下にある **Create queue** ボタンをクリックして、キューの作成を完了しましょう。

## Volcano のインストール

Kubernetes クラスターに Volcano をインストールするには、[公式インストールガイド](https://volcano.sh/en/docs/installation/) に従ってください。

## Launch エージェント のデプロイ

キューを作成したので、次に Launch エージェントをデプロイして、キューからジョブを取得・実行できるようにします。一番簡単な方法は、[W&B の公式 `helm-charts` リポジトリの `launch-agent` チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)を使う方法です。README のインストール手順に従い、Kubernetes クラスターにチャートをインストールしてください。また、先ほど作成したキューをエージェントがポーリングできるように必ず設定してください。

## トレーニングジョブの作成

Volcano の pytorch プラグインは、`MASTER_ADDR`、`RANK`、`WORLD_SIZE` など、pytorch DDP に必要な環境変数を自動で設定します。お使いの pytorch コードが DDP を正しく利用していれば、そのまま分散トレーニングが動作します。DDP の詳細については [pytorch のドキュメント](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) をご参照ください。

{{% alert %}}
Volcano の pytorch プラグインは、[PyTorch Lightning の `Trainer` を利用したマルチノードトレーニング](https://lightning.ai/docs/pytorch/stable/common/trainer.html#num-nodes)にも対応しています。
{{% /alert %}}

## Launch

キューとクラスターの準備ができたら、いよいよ分散トレーニングをローンチします。まずは、[このジョブ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest)を使って、volcano の pytorch プラグインでランダムデータにマルチレイヤパーセプトロンをトレーニングする例を見てみましょう。ジョブのソースコードは[こちら](https://github.com/wandb/launch-jobs/tree/main/jobs/distributed_test)にあります。

このジョブをローンチするには、[ジョブのページ](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest)にアクセスし、画面右上の **Launch** ボタンをクリックします。ローンチするキューの選択を求められます。

{{< img src="/images/launch/launching_multinode_job.png" alt="Multi-node job launch" >}}

1. 好きなようにジョブのパラメータを設定します
2. 先ほど作成したキューを選択します
3. **Resource config** セクション内の volcano job を編集して、ジョブのパラメータを変更できます。たとえば、`worker` タスクの `replicas` フィールドを変えることでワーカー数を調整できます
4. **Launch** をクリック

ジョブの進行状況は W&B UI から確認・監視でき、必要に応じて停止も可能です。