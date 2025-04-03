---
title: 'Tutorial: Set up W&B Launch on SageMaker'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-sagemaker
    parent: set-up-launch
url: guides/launch/setup-launch-sagemaker
---

W&B の Launch を使用すると、提供された、またはカスタムのアルゴリズムを使用して、Amazon SageMaker に Launch ジョブを送信し、SageMaker プラットフォームで機械学習 モデルをトレーニングできます。SageMaker は、コンピューティングリソースの起動と解放を行うため、EKS クラスターを持たない Teams にとって良い選択肢となります。

Amazon SageMaker に接続された W&B Launch キューに送信された Launch ジョブは、[CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) を使用して SageMaker Training ジョブとして実行されます。Launch キューの設定を使用して、`CreateTrainingJob` API に送信される引数を制御します。

Amazon SageMaker は、[Docker イメージを使用して Training ジョブを実行します](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html)。SageMaker によってプルされるイメージは、Amazon Elastic Container Registry (ECR) に保存する必要があります。これは、トレーニングに使用するイメージが ECR に保存されている必要があることを意味します。

{{% alert %}}
この ガイド では、SageMaker Training ジョブを実行する方法について説明します。Amazon SageMaker での推論のために Models をデプロイする方法については、[この Launch ジョブの例](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints) を参照してください。
{{% /alert %}}

## 前提条件

開始する前に、次の前提条件を満たしていることを確認してください。

* [Launch エージェント に Docker イメージを構築させるかどうかを決定します。]({{< relref path="#decide-if-you-want-the-launch-agent-to-build-a-docker-images" lang="ja" >}})
* [AWS リソースをセットアップし、S3、ECR、および Sagemaker IAM ロールに関する情報を収集します。]({{< relref path="#set-up-aws-resources" lang="ja" >}})
* [Launch エージェント の IAM ロールを作成します。]({{< relref path="#create-an-iam-role-for-launch-agent" lang="ja" >}})

### Launch エージェント に Docker イメージを構築させるかどうかを決定します。

W&B Launch エージェント に Docker イメージを構築させるかどうかを決定します。次の 2 つのオプションから選択できます。

* Launch エージェント が Docker イメージを構築し、イメージを Amazon ECR にプッシュして、[SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) ジョブを送信できるようにします。このオプションは、機械学習 エンジニアがトレーニング コードを迅速に反復処理する際に、ある程度の簡素化をもたらすことができます。
* Launch エージェント は、トレーニング スクリプトまたは推論スクリプトを含む既存の Docker イメージを使用します。このオプションは、既存の CI システムとうまく連携します。このオプションを選択した場合は、Docker イメージを Amazon ECR のコンテナー レジストリに手動でアップロードする必要があります。

### AWS リソースのセットアップ

優先する AWS リージョンで次の AWS リソースが構成されていることを確認してください。

1. コンテナー イメージを保存する [ECR リポジトリ](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html)。
2. SageMaker Training ジョブの入出力を保存する 1 つ以上の [S3 バケット](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)。
3. SageMaker が Training ジョブを実行し、Amazon ECR および Amazon S3 とやり取りすることを許可する Amazon SageMaker の IAM ロール。

これらのリソースの ARN をメモしておきます。[Launch キューの設定]({{< relref path="#configure-launch-queue-for-sagemaker" lang="ja" >}})を定義する際に、ARN が必要になります。

### Launch エージェント の IAM ポリシーを作成する

1. AWS の IAM 画面から、新しいポリシーを作成します。
2. JSON ポリシー エディターに切り替え、ユースケースに基づいて次のポリシーを貼り付けます。`<>` で囲まれた値を独自の値に置き換えます。

{{< tabpane text=true >}}
{{% tab "エージェント が構築済みの Docker イメージを送信する" %}}
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
{{% tab "エージェント が Docker イメージを構築して送信する" %}}
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

3. [**Next**] をクリックします。
4. ポリシーに名前と説明を付けます。
5. [**Create policy**] をクリックします。

### Launch エージェント の IAM ロールを作成する

Launch エージェント に Amazon SageMaker Training ジョブを作成する権限が必要です。次の手順に従って、IAM ロールを作成します。

1. AWS の IAM 画面から、新しいロールを作成します。
2. [**Trusted Entity**] で、[**AWS Account**] (または組織のポリシーに適した別のオプション) を選択します。
3. 権限画面をスクロールして、上記で作成したポリシー名を選択します。
4. ロールに名前と説明を付けます。
5. [**Create role**] を選択します。
6. ロールの ARN をメモします。Launch エージェント を設定する際に、ARN を指定します。

IAM ロールの作成方法の詳細については、[AWS Identity and Access Management のドキュメント](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html)を参照してください。

{{% alert %}}
* Launch エージェント にイメージを構築させる場合は、追加で必要な権限について、[エージェント の高度な設定]({{< relref path="./setup-agent-advanced.md" lang="ja" >}})を参照してください。
* SageMaker キューの `kms:CreateGrant` 権限は、関連付けられた ResourceConfig に VolumeKmsKeyId が指定されていて、関連付けられたロールにこのアクションを許可するポリシーがない場合にのみ必要です。
{{% /alert %}}

## SageMaker の Launch キューを設定する

次に、SageMaker をコンピューティング リソースとして使用するキューを W&B アプリ で作成します。

1. [Launch アプリ](https://wandb.ai/launch)に移動します。
3. [**Create Queue**] ボタンをクリックします。
4. キューを作成する [**Entity**] を選択します。
5. [**Name**] フィールドにキューの名前を入力します。
6. [**Resource**] として [**SageMaker**] を選択します。
7. [**Configuration**] フィールド内で、SageMaker ジョブに関する情報を提供します。デフォルトでは、W&B は YAML および JSON の `CreateTrainingJob` リクエスト本文を生成します。
```json
{
  "RoleArn": "<必須>", 
  "ResourceConfig": {
      "InstanceType": "ml.m4.xlarge",
      "InstanceCount": 1,
      "VolumeSizeInGB": 2
  },
  "OutputDataConfig": {
      "S3OutputPath": "<必須>"
  },
  "StoppingCondition": {
      "MaxRuntimeInSeconds": 3600
  }
}
```
少なくとも以下を指定する必要があります。

- `RoleArn`: SageMaker 実行 IAM ロールの ARN ([前提条件]({{< relref path="#prerequisites" lang="ja" >}})を参照)。Launch **エージェント** IAM ロールと混同しないようにしてください。
- `OutputDataConfig.S3OutputPath`: SageMaker の出力が保存される Amazon S3 URI。
- `ResourceConfig`: リソース設定に必要な仕様。リソース設定のオプションは[こちら](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceConfig.html)に概説されています。
- `StoppingCondition`: Training ジョブの停止条件に必要な仕様。オプションは[こちら](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)に概説されています。
7. [**Create Queue**] ボタンをクリックします。

## Launch エージェント を設定する

次のセクションでは、エージェント をデプロイできる場所と、デプロイ場所に基づいて エージェント を構成する方法について説明します。

[Amazon SageMaker の Launch エージェント をデプロイする方法には、いくつかのオプション]({{< relref path="#decide-where-to-run-the-launch-agent" lang="ja" >}})があります。ローカル マシン、EC2 インスタンス、または EKS クラスターです。[エージェント をデプロイする場所に基づいて、Launch エージェント を適切に構成]({{< relref path="#configure-a-launch-agent" lang="ja" >}})します。

### Launch エージェント を実行する場所を決定する

本番環境のワークロードや、既に EKS クラスターをお持ちのお客様には、この Helm チャートを使用して、Launch エージェント を EKS クラスターにデプロイすることをお勧めします。

現在の EKS クラスターを使用しない本番環境のワークロードの場合、EC2 インスタンスは優れたオプションです。Launch エージェント インスタンスは常に実行され続けますが、エージェント には `t2.micro` サイズの EC2 インスタンス以上のものは必要ありません。これは比較的安価です。

実験的なユースケースや個人のユースケースの場合、ローカル マシンで Launch エージェント を実行すると、すばやく開始できます。

ユースケースに基づいて、次のタブに記載されている手順に従って、Launch エージェント を適切に構成してください。
{{< tabpane text=true >}}
{{% tab "EKS" %}}
W&B は、[W&B 管理の Helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)を使用して、EKS クラスターに エージェント をインストールすることを強くお勧めします。
{{% /tab %}}
{{% tab "EC2" %}}
Amazon EC2 ダッシュボードに移動し、次の手順を実行します。

1. [**Launch instance**] をクリックします。
2. [**Name**] フィールドに名前を入力します。必要に応じて、タグを追加します。
2. [**Instance type**] で、EC2 コンテナーのインスタンス タイプを選択します。1 vCPU と 1 GiB のメモリを超えるものは必要ありません (たとえば、t2.micro)。
3. [**Key pair (login)**] フィールド内で、組織のキー ペアを作成します。このキー ペアを使用して、後の手順で SSH クライアントを使用して[EC2 インスタンスに接続](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html)します。
2. [**Network settings**] 内で、組織に適したセキュリティ グループを選択します。
3. [**Advanced details**] を展開します。[**IAM instance profile**] で、上記で作成した Launch エージェント IAM ロールを選択します。
2. [**Summary**] フィールドを確認します。正しい場合は、[**Launch instance**] を選択します。

AWS の EC2 ダッシュボードの左側のパネルにある [**Instances**] に移動します。作成した EC2 インスタンスが実行されていることを確認します ([**Instance state**] 列を参照)。EC2 インスタンスが実行されていることを確認したら、ローカル マシンのターミナルに移動して、次の手順を実行します。

1. [**Connect**] を選択します。
2. [**SSH client**] タブを選択し、概要が示されている手順に従って EC2 インスタンスに接続します。
3. EC2 インスタンス内で、次のパッケージをインストールします。
```bash
sudo yum install python311 -y && python3 -m ensurepip --upgrade && pip3 install wandb && pip3 install wandb[launch]
```
4. 次に、EC2 インスタンス内で Docker をインストールして起動します。
```bash
sudo yum update -y && sudo yum install -y docker python3 && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -a -G docker ec2-user

newgrp docker
```

これで、Launch エージェント の構成に進むことができます。

{{% /tab %}}
{{% tab "ローカル マシン" %}}

`~/.aws/config` および `~/.aws/credentials` にある AWS 構成ファイルを使用して、ローカル マシンでポーリングする エージェント にロールを関連付けます。前の手順で Launch エージェント 用に作成した IAM ロール ARN を指定します。
 
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

セッション トークンには、関連付けられているプリンシパルに応じて、[最大長](https://docs.aws.amazon.com/cli/latest/reference/sts/get-session-token.html#description)が 1 時間または 3 日であることに注意してください。
{{% /tab %}}
{{< /tabpane >}}

### Launch エージェント を構成する
YAML 構成ファイル `launch-config.yaml` を使用して Launch エージェント を構成します。

デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` で構成ファイルを確認します。必要に応じて、`-c` フラグを使用して Launch エージェント をアクティブ化するときに、別のディレクトリーを指定できます。

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

次に、`wandb launch-agent` で エージェント を開始します。

## (オプション) Launch ジョブ Docker イメージを Amazon ECR にプッシュする

{{% alert %}}
このセクションは、Launch エージェント がトレーニング ロジックまたは推論ロジックを含む既存の Docker イメージを使用する場合にのみ適用されます。[Launch エージェント の動作方法には 2 つのオプションがあります。]({{< relref path="#decide-if-you-want-the-launch-agent-to-build-a-docker-images" lang="ja" >}})
{{% /alert %}}

Launch ジョブを含む Docker イメージを Amazon ECR リポジトリにアップロードします。イメージベースのジョブを使用している場合は、新しい Launch ジョブを送信する前に、Docker イメージが ECR レジストリに存在する必要があります。
```
