---
title: 'Tutorial: Set up W&B Launch on SageMaker'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-sagemaker
    parent: set-up-launch
url: guides/launch/setup-launch-sagemaker
---

W&B の Launch を使用すると、提供された、またはカスタムのアルゴリズムを使用して機械学習モデルをトレーニングするために、Amazon SageMaker に Launch ジョブを送信できます。SageMaker はコンピューティングリソースの起動と解放を処理するため、EKS クラスターを持たない Team にとって良い選択肢となります。

Amazon SageMaker に接続された W&B Launch queue に送信された Launch ジョブは、[CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) を使用した SageMaker Training Jobs として実行されます。Launch queue の設定を使用して、`CreateTrainingJob` API に送信される引数を制御します。

Amazon SageMaker は、[Docker イメージを使用して Training Jobs を実行します](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html)。SageMaker によって pull されるイメージは、Amazon Elastic Container Registry (ECR) に保存する必要があります。つまり、Training に使用するイメージは ECR に保存する必要があります。

{{% alert %}}
この ガイド では、SageMaker Training Jobs を実行する方法について説明します。Amazon SageMaker での推論のために model をデプロイする方法については、[この Launch ジョブの例](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints) を参照してください。
{{% /alert %}}

## 前提条件

始める前に、次の前提条件を満たしていることを確認してください。

* [Launch エージェント に Docker イメージを構築させるかどうかを決定します。]({{< relref path="#decide-if-you-want-the-launch-agent-to-build-a-docker-images" lang="ja" >}})
* [AWS リソースをセットアップし、S3、ECR、および SageMaker IAM ロールに関する情報を収集します。]({{< relref path="#set-up-aws-resources" lang="ja" >}})
* [Launch エージェント の IAM ロールを作成します。]({{< relref path="#create-an-iam-role-for-launch-agent" lang="ja" >}})

### Launch エージェント に Docker イメージを構築させるかどうかを決定する

W&B Launch エージェント に Docker イメージを構築させるかどうかを決定します。次の 2 つのオプションから選択できます。

* Launch エージェント が Docker イメージを構築し、イメージを Amazon ECR に push し、[SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) ジョブを送信することを許可します。このオプションは、ML エンジニアが Training コードを迅速に反復処理する際に、ある程度の簡素化を提供できます。
* Launch エージェント は、Training または推論スクリプトを含む既存の Docker イメージを使用します。このオプションは、既存の CI システムとうまく連携します。このオプションを選択する場合は、Docker イメージを Amazon ECR のコンテナレジストリに手動でアップロードする必要があります。

### AWS リソースのセットアップ

希望する AWS リージョンで次の AWS リソースが構成されていることを確認してください。

1. コンテナイメージを保存するための[ECR リポジトリ](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html)。
2. SageMaker Training ジョブの入力と出力を保存するための 1 つ以上の[S3 バケット](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)。
3. SageMaker が Training ジョブを実行し、Amazon ECR および Amazon S3 とやり取りすることを許可する Amazon SageMaker の IAM ロール。

これらのリソースの ARN をメモしておきます。[Launch queue の設定]({{< relref path="#configure-launch-queue-for-sagemaker" lang="ja" >}})を定義する際に、ARN が必要になります。

### Launch エージェント の IAM ポリシーを作成する

1. AWS の IAM 画面から、新しいポリシーを作成します。
2. JSON ポリシーエディタに切り替え、ユースケースに基づいて次のポリシーを貼り付けます。`<>` で囲まれた 値 は、自分の 値 に置き換えます。

{{< tabpane text=true >}}
{{% tab "エージェント は事前に構築された Docker イメージを送信します" %}}
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
{{% tab "エージェント は Docker イメージを構築して送信します" %}}
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

### Launch エージェント の IAM ロールを作成する

Launch エージェント に Amazon SageMaker Training Jobs を作成する権限が必要です。次の手順に従って、IAM ロールを作成します。

1. AWS の IAM 画面から、新しいロールを作成します。
2. **Trusted Entity** で、**AWS Account** (または、組織のポリシーに適した別のオプション) を選択します。
3. 許可画面をスクロールして、上記で作成したポリシー名を選択します。
4. ロールに名前と説明を付けます。
5. **Create role** を選択します。
6. ロールの ARN をメモします。Launch エージェント を設定するときに、ARN を指定します。

IAM ロールの作成方法の詳細については、[AWS Identity and Access Management ドキュメント](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html)を参照してください。

{{% alert %}}
* Launch エージェント にイメージを構築させる場合は、[Advanced agent set up]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) を参照して、追加の権限を確認してください。
* SageMaker queue の `kms:CreateGrant` 権限は、関連付けられた ResourceConfig に VolumeKmsKeyId が指定されていて、関連付けられたロールにこのアクションを許可するポリシーがない場合にのみ必要です。
{{% /alert %}}

## SageMaker の Launch queue を設定する

次に、SageMaker をコンピューティングリソースとして使用する queue を W&B App に作成します。

1. [Launch App](https://wandb.ai/launch) に移動します。
3. **Create Queue** ボタンをクリックします。
4. queue を作成する **Entity** を選択します。
5. **Name** フィールドに queue の名前を入力します。
6. **Resource** として **SageMaker** を選択します。
7. **Configuration** フィールドに、SageMaker ジョブに関する情報を入力します。デフォルトでは、W&B は YAML および JSON の `CreateTrainingJob` リクエスト本文を入力します。
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
少なくとも、以下を指定する必要があります。

- `RoleArn` : SageMaker 実行 IAM ロールの ARN ([前提条件]({{< relref path="#prerequisites" lang="ja" >}})を参照)。Launch **エージェント** IAM ロールと混同しないでください。
- `OutputDataConfig.S3OutputPath` : SageMaker の出力が保存される場所を指定する Amazon S3 URI。
- `ResourceConfig`: リソース構成の必要な仕様。リソース構成のオプションは[こちら](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceConfig.html)に記載されています。
- `StoppingCondition`: Training ジョブの停止条件に必要な仕様。オプションは[こちら](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)に記載されています。
7. **Create Queue** ボタンをクリックします。

## Launch エージェント を設定する

次のセクションでは、エージェント をデプロイできる場所と、デプロイ場所に基づいてエージェント を設定する方法について説明します。

[Amazon SageMaker の Launch エージェント のデプロイ方法にはいくつかのオプションがあります]({{< relref path="#decide-where-to-run-the-launch-agent" lang="ja" >}}): ローカルマシン、EC2 インスタンス、または EKS クラスター。[エージェント をデプロイする場所に基づいて、Launch エージェント を適切に設定してください]({{< relref path="#configure-a-launch-agent" lang="ja" >}})。

### Launch エージェント を実行する場所を決定する

本番環境のワークロード、およびすでに EKS クラスターを持っている顧客の場合、W&B はこの Helm チャートを使用して、Launch エージェント を EKS クラスターにデプロイすることをお勧めします。

現在の EKS クラスターがない本番環境のワークロードの場合、EC2 インスタンスは良い選択肢です。Launch エージェント インスタンスは常に実行されていますが、エージェント は比較的安価な `t2.micro` サイズの EC2 インスタンス以上のものを必要としません。

実験的またはソロのユースケースの場合、ローカルマシンで Launch エージェント を実行すると、すばやく開始できます。

ユースケースに基づいて、次の tab に記載されている手順に従って、Launch エージェント を適切に設定してください。
{{< tabpane text=true >}}
{{% tab "EKS" %}}
W&B は、[W&B マネージド Helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)を使用して、EKS クラスターに エージェント をインストールすることを強く推奨します。
{{% /tab %}}
{{% tab "EC2" %}}
Amazon EC2 ダッシュボードに移動し、次の手順を実行します。

1. **Launch instance** をクリックします。
2. **Name** フィールドに名前を入力します。必要に応じて、tag を追加します。
2. **Instance type** から、EC2 コンテナのインスタンスタイプを選択します。1vCPU と 1GiB のメモリ (t2.micro など) 以上は必要ありません。
3. **Key pair (login)** フィールド内で、組織のキーペアを作成します。このキーペアを使用して、後の手順で SSH クライアントで[EC2 インスタンスに接続](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html)します。
2. **Network settings** 内で、組織に適したセキュリティグループを選択します。
3. **Advanced details** を展開します。**IAM instance profile** で、上記で作成した Launch エージェント IAM ロールを選択します。
2. **Summary** フィールドを確認します。正しい場合は、**Launch instance** を選択します。

AWS の EC2 ダッシュボードの左側のパネルにある **Instances** に移動します。作成した EC2 インスタンスが実行されていることを確認します (**Instance state** 列を参照)。EC2 インスタンスが実行されていることを確認したら、ローカルマシンのターミナルに移動して、次の操作を実行します。

1. **Connect** を選択します。
2. **SSH client** タブを選択し、概要が示されている手順に従って EC2 インスタンスに接続します。
3. EC2 インスタンス内で、次のパッケージをインストールします。
```bash
sudo yum install python311 -y && python3 -m ensurepip --upgrade && pip3 install wandb && pip3 install wandb[launch]
```
4. 次に、EC2 インスタンス内で Docker をインストールして起動します。
```bash
sudo yum update -y && sudo yum install -y docker python3 && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -a -G docker ec2-user

newgrp docker
```

これで、Launch エージェント の設定に進むことができます。

{{% /tab %}}
{{% tab "ローカルマシン" %}}

`~/.aws/config` および `~/.aws/credentials` にある AWS 構成ファイルを使用して、ローカルマシンでポーリングしている エージェント にロールを関連付けます。前の手順で Launch エージェント 用に作成した IAM ロール ARN を指定します。
 
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

セッショントークンには、関連付けられているプリンシパルに応じて、[最大長](https://docs.aws.amazon.com/cli/latest/reference/sts/get-session-token.html#description)が 1 時間または 3 日であることに注意してください。
{{% /tab %}}
{{< /tabpane >}}

### Launch エージェント を設定する
YAML 構成ファイル `launch-config.yaml` を使用して、Launch エージェント を設定します。

デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` で構成ファイルを確認します。`-c` フラグを使用して Launch エージェント をアクティブ化するときに、別のディレクトリーをオプションで指定できます。

次の YAML スニペットは、コア構成 エージェント オプションを指定する方法を示しています。

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

次に、`wandb launch-agent` で エージェント を起動します。

## (オプション) Launch ジョブの Docker イメージを Amazon ECR に push する

{{% alert %}}
このセクションは、Launch エージェント が Training または推論ロジックを含む既存の Docker イメージを使用する場合にのみ適用されます。[Launch エージェント の動作方法には 2 つのオプションがあります。]({{< relref path="#decide-if-you-want-the-launch-agent-to-build-a-docker-images" lang="ja" >}})
{{% /alert %}}

Launch ジョブを含む Docker イメージを Amazon ECR リポジトリにアップロードします。イメージベースのジョブを使用している場合は、新しい Launch ジョブを送信する前に、Docker イメージを ECR レジストリに格納する必要があります。
```