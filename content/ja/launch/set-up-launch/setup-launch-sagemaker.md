---
title: 'チュートリアル: W&B Launch を SageMaker でセットアップする'
menu:
  launch:
    identifier: setup-launch-sagemaker
    parent: set-up-launch
url: guides/launch/setup-launch-sagemaker
---

W&B Launch を使うことで、SageMaker プラットフォーム上で提供されたアルゴリズムやカスタムアルゴリズムを用いて機械学習モデルのトレーニングジョブを Amazon SageMaker に送信できます。SageMaker は計算リソースの起動や開放を自動で管理してくれるので、EKS クラスターを持たないチームにも適した選択肢です。

W&B Launch キューから Amazon SageMaker に送信された Launch ジョブは、[CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) を使って SageMaker Training Job として実行されます。Launch キューの設定で、`CreateTrainingJob` API へ送信される引数を制御できます。

Amazon SageMaker は[ Docker イメージでトレーニングジョブを実行します](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html)。SageMaker が取得するイメージは Amazon Elastic Container Registry (ECR) に保存されている必要があります。つまり、トレーニングに使うイメージは ECR 上に保存されていなければなりません。

{{% alert %}}
このガイドでは、SageMaker Training Job の実行方法を説明します。推論用モデルを Amazon SageMaker へデプロイする方法については、[こちらの Launch ジョブ例](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints) をご覧ください。
{{% /alert %}}


## 前提条件

始める前に、以下の前提条件を満たしていることを確認してください。

* [Launch agent に Docker イメージのビルドを任せるかを決めます。]({{< relref "#decide-if-you-want-the-launch-agent-to-build-a-docker-images" >}})
* [AWS リソースのセットアップと、S3・ECR・SageMaker IAM ロールに関する情報の取得。]({{< relref "#set-up-aws-resources" >}})
* [Launch agent 用の IAM ロールを作成します]({{< relref "#create-an-iam-role-for-launch-agent" >}})。

### Launch agent に Docker イメージのビルドを任せるか決める

W&B Launch agent に Docker イメージのビルドを任せるか選択します。以下 2 通りの方法から選べます：

* Launch agent に Docker イメージをビルド・ECR へプッシュ、その上で [SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) ジョブを提出してもらう方法。ML エンジニアがトレーニングコードを素早く改良したい場合に、シンプルに実現できます。
* 既存の Docker イメージを用いて、トレーニングや推論用スクリプトを含める方法。既存の CI システムとうまく連携します。この場合は、ご自身で Docker イメージを Amazon ECR のコンテナレジストリへアップロードしてください。


### AWS リソースのセットアップ

希望する AWS リージョンで、次の AWS リソースが設定されていることを確認してください:

1. コンテナイメージを保存するための [ECR リポジトリ](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html)。
2. SageMaker Training ジョブの入力および出力用の [S3 バケット](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)（1つ以上）。
3. SageMaker Training ジョブ作成や、Amazon ECR・Amazon S3 との連携操作を許可する Amazon SageMaker 用の IAM ロール。

これらリソースの ARN を控えておいてください。[Launch キュー設定]({{< relref "#configure-launch-queue-for-sagemaker" >}}) で必要になります。

### Launch agent 用 IAM ポリシーの作成

1. AWS の IAM 画面から新しいポリシーを作成します。
2. JSON ポリシーエディタを開き、次のポリシーをユースケースに合わせて貼り付けます。`<>` で囲まれた値はご自身の値に置き換えてください。

{{< tabpane text=true >}}
{{% tab "Agent submits pre-built Docker image" %}}
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "logs:DescribeLogStreams",
          "SageMaker:AddTags",
          "SageMaker:CreateTrainingJob",
          "SageMaker:DescribeTrainingJob"
        ],
        "Resource": "arn:aws:sagemaker:<region>:<account-id>:*"
      },
      {
        "Effect": "Allow",
        "Action": "iam:PassRole",
        "Resource": "arn:aws:iam::<account-id>:role/<RoleArn-from-queue-config>"
      },
    {
        "Effect": "Allow",
        "Action": "kms:CreateGrant",
        "Resource": "<ARN-OF-KMS-KEY>",
        "Condition": {
          "StringEquals": {
            "kms:ViaService": "SageMaker.<region>.amazonaws.com",
            "kms:GrantIsForAWSResource": "true"
          }
        }
      }
    ]
  }
  ```
{{% /tab %}}
{{% tab "Agent builds and submits Docker image" %}}
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "logs:DescribeLogStreams",
          "SageMaker:AddTags",
          "SageMaker:CreateTrainingJob",
          "SageMaker:DescribeTrainingJob"
        ],
        "Resource": "arn:aws:sagemaker:<region>:<account-id>:*"
      },
      {
        "Effect": "Allow",
        "Action": "iam:PassRole",
        "Resource": "arn:aws:iam::<account-id>:role/<RoleArn-from-queue-config>"
      },
       {
      "Effect": "Allow",
      "Action": [
        "ecr:CreateRepository",
        "ecr:UploadLayerPart",
        "ecr:PutImage",
        "ecr:CompleteLayerUpload",
        "ecr:InitiateLayerUpload",
        "ecr:DescribeRepositories",
        "ecr:DescribeImages",
        "ecr:BatchCheckLayerAvailability",
        "ecr:BatchDeleteImage"
      ],
      "Resource": "arn:aws:ecr:<region>:<account-id>:repository/<repository>"
    },
    {
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
        "Effect": "Allow",
        "Action": "kms:CreateGrant",
        "Resource": "<ARN-OF-KMS-KEY>",
        "Condition": {
          "StringEquals": {
            "kms:ViaService": "SageMaker.<region>.amazonaws.com",
            "kms:GrantIsForAWSResource": "true"
          }
        }
      }
    ]
  }
  ```
{{% /tab %}}
{{< /tabpane >}}

3. **Next** をクリックします。
4. ポリシーに名前と説明を付けます。
5. **Create policy** をクリックします。


### Launch agent 用 IAM ロールの作成

Launch agent には Amazon SageMaker のトレーニングジョブ作成権限が必要です。以下の手順で IAM ロールを作成してください。

1. AWS の IAM 画面から新しいロールを作成します。
2. **Trusted Entity** で **AWS Account**（または自社ポリシーに合致する他のオプション）を選択します。
3. パーミッション画面をスクロールして、先ほど作成したポリシー名を選択。
4. ロールに名前と説明を付けてください。
5. **Create role** を選択。
6. ロールの ARN を控えてください。後ほど launch agent 設定時に必要です。

IAM ロールの作成手順については、[AWS Identity and Access Management ドキュメント](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html) を参照してください。

{{% alert %}}
* Launch agent でイメージビルドを行う場合、追加権限については [高度な agent セットアップ]({{< relref "./setup-agent-advanced.md" >}}) を参照してください。
* SageMaker キュー用の `kms:CreateGrant` パーミッションは、関連する ResourceConfig に VolumeKmsKeyId を指定しており、関連ロールのポリシーでこの操作が許可されていない場合のみ必要です。
{{% /alert %}}



## SageMaker 用の Launch キュー設定

次に、W&B App 上で SageMaker を計算リソースに指定したキューを作成します。

1. [Launch App](https://wandb.ai/launch) へ移動します。
3. **Create Queue** ボタンをクリックします。
4. 作成したい **Entity** を選択します。
5. **Name** フィールドにキュー名を入力します。
6. **Resource** で **SageMaker** を選択します。
7. **Configuration** フィールドには SageMaker ジョブの情報を入力します。デフォルトで、W&B により YAML および JSON の `CreateTrainingJob` リクエストボディが自動入力されます。
```json
{
  "RoleArn": "<REQUIRED>", 
  "ResourceConfig": {
      "InstanceType": "ml.m4.xlarge",
      "InstanceCount": 1,
      "VolumeSizeInGB": 2
  },
  "OutputDataConfig": {
      "S3OutputPath": "<REQUIRED>"
  },
  "StoppingCondition": {
      "MaxRuntimeInSeconds": 3600
  }
}
```
最低限、以下の項目は指定必須です：

- `RoleArn` : SageMaker 実行用 IAM ロールの ARN（[前提条件]({{< relref "#prerequisites" >}}) を参照）。Launch **agent** 用 IAM ロールと混同しないようご注意ください。
- `OutputDataConfig.S3OutputPath` : SageMaker の出力保存先となる Amazon S3 URI。
- `ResourceConfig` ：リソース設定。オプション詳細は[こちら](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceConfig.html)。
- `StoppingCondition` ：トレーニングジョブ用の停止条件。オプションは[こちら](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)。
7. **Create Queue** ボタンをクリックしてください。


## Launch agent のセットアップ

このセクションでは、agent のデプロイ先の選択と、その場所ごとに適した agent 設定方法について説明します。

[Amazon SageMaker 用 Launch agent のデプロイ先には複数オプションがあります]({{< relref "#decide-where-to-run-the-launch-agent" >}})：ローカルマシン、EC2 インスタンス、EKS クラスターなど。デプロイ先に合わせて [launch agent を適切に設定してください]({{< relref "#configure-a-launch-agent" >}})。


### Launch agent の実行場所を決める

本番運用ワークロードや EKS クラスターを保有しているカスタマーには、W&B 提供の Helm チャートを使って Launch agent を EKS クラスターにデプロイすることを推奨します。

EKS クラスターが無い場合は EC2 インスタンスの利用が良いでしょう。agent インスタンスは常時起動となりますが、`t2.micro` のようなリーズナブルな EC2 インスタンスタイプで十分動作します。

試行や個人利用には、ローカルマシン上で Launch agent を実行するのが素早い方法です。

ご自身のユースケースに合った項目を選び、タブに従って launch agent の設定方法を確認してください。
{{< tabpane text=true >}}
{{% tab "EKS" %}}
W&B では [W&B 管理の helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使った EKS クラスターへの agent インストールを強く推奨します。
{{% /tab %}}
{{% tab "EC2" %}}
Amazon EC2 ダッシュボードにアクセスし、次の手順を実行してください。

1. **Launch instance** をクリックします。
2. **Name** フィールドに名前を入力。必要に応じてタグを追加します。
2. **Instance type** から EC2 用のタイプを選択します。1vCPU および 1GiB メモリがあれば十分です（例: t2.micro）。
3. **Key pair (login)** フィールドで組織用のキーペアを作成します。後ほど、このキーペアを使って [SSH クライアントで EC2 インスタンスへ接続](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html) する必要があります。
2. **Network settings** で組織に合ったセキュリティグループを選択します。
3. **Advanced details** を展開し、**IAM instance profile** で先ほど作成した Launch agent 用 IAM ロールを選択します。
2. **Summary** フィールドで内容を確認し、問題なければ **Launch instance** を選択。

AWS の EC2 ダッシュボード左パネルの **Instances** を開き、作成した EC2 インスタンスが稼働中であることを確認してください（**Instance state** 列参照）。インスタンス稼働を確認したら、ローカルマシンのターミナルから次を行います:

1. **Connect** を選択。
2. **SSH client** タブを選び、記載の手順で EC2 インスタンスに接続します。
3. EC2 内で次のパッケージをインストールします。
```bash
sudo yum install python311 -y && python3 -m ensurepip --upgrade && pip3 install wandb && pip3 install wandb[launch]
```
4. 次に、EC2 内で Docker をインストールし起動します。
```bash
sudo yum update -y && sudo yum install -y docker python3 && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -a -G docker ec2-user

newgrp docker
```

これで、Launch agent の設定に進むことができます。

{{% /tab %}}
{{% tab "Local machine" %}}

ローカルマシンで agent を動かす場合は、`~/.aws/config` および `~/.aws/credentials` ファイルで role を紐付けます。先ほど作成した Launch agent 用 IAM ロールの ARN を記載します。

```yaml title="~/.aws/config"
[profile SageMaker-agent]
role_arn = arn:aws:iam::<account-id>:role/<agent-role-name>
source_profile = default                                                                   
```

```yaml title="~/.aws/credentials"
[default]
aws_access_key_id=<access-key-id>
aws_secret_access_key=<secret-access-key>
aws_session_token=<session-token>
```

なお、セッショントークンには [最大有効長](https://docs.aws.amazon.com/cli/latest/reference/sts/get-session-token.html#description) （1 時間 or 3 日）があることにご注意ください。
{{% /tab %}}
{{< /tabpane >}}


### Launch agent の設定

YAML 設定ファイル `launch-config.yaml` で launch agent を設定します。

デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` のファイルを参照します。`-c` フラグを利用して、別のディレクトリーを指定することも可能です。

以下の YAML 例は、コアとなる config agent オプションの記載例です。

```yaml title="launch-config.yaml"
max_jobs: -1
queues:
  - <queue-name>
environment:
  type: aws
  region: <your-region>
registry:
  type: ecr
  uri: <ecr-repo-arn>
builder: 
  type: docker

```

`wandb launch-agent` で agent を起動します。


 ## （任意）Launch job 用 Docker イメージを Amazon ECR へプッシュ

{{% alert %}}
このセクションは、ご自身の Launch agent で既存 Docker イメージ（トレーニングや推論ロジックを含んだもの）を使う場合のみ該当します。[Launch agent の挙動には 2 通りの方法があります。]({{< relref "#decide-if-you-want-the-launch-agent-to-build-a-docker-images" >}})  
{{% /alert %}}

Launch job を含む Docker イメージを、Amazon ECR リポジトリへアップロードしてください。イメージベースのジョブを利用する場合、Launch ジョブ申請前にこのイメージを ECR レジストリへ登録しておく必要があります。