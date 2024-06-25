---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Set up for SageMaker

W&B Launch を使用して、SageMaker プラットフォームで提供されているアルゴリズムやカスタムアルゴリズムを用いて、機械学習モデルをトレーニングするためのジョブを Amazon SageMaker に送信できます。SageMaker はコンピュートリソースの起動とリリースを担当するため、EKS クラスターがないチームにとって良い選択となることがあります。

Amazon SageMaker に接続された W&B Launch キューに送信された Launch ジョブは、[CreateTrainingJob API](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_CreateTrainingJob.html) を使用して SageMaker Training Jobs として実行されます。Launch キューの設定を使用して `CreateTrainingJob` API に送信される引数を制御します。

Amazon SageMaker は[Docker イメージを使用してトレーニングジョブを実行](https://docs.aws.amazon.com/SageMaker/latest/dg/your-algorithms-training-algo-dockerfile.html)します。SageMakerにプルされるイメージは Amazon Elastic Container Registry (ECR) に保存されている必要があります。つまり、トレーニングに使用するイメージは ECR に保存されていなければなりません。

:::note
このガイドでは、SageMaker トレーニングジョブの実行方法を示します。Amazon SageMaker で推論用のモデルをデプロイする方法については、[この例の Launch ジョブ](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints)を参照してください。
:::

## 必要条件

始める前に、以下の前提条件を満たしていることを確認してください。

* [Launch エージェントが Docker イメージを構築するかどうかを決定する。](#decide-if-you-want-the-launch-agent-to-build-a-docker-images)
* [AWS リソースをセットアップし、S3、ECR、Sagemaker IAM ロールに関する情報を収集する。](#set-up-aws-resources)
* [Launch エージェントのために IAM ロールを作成する](#create-an-iam-role-for-launch-agent)。

### Launch エージェントが Docker イメージを構築するかどうかを決定する

W&B Launch エージェントが Docker イメージを構築するかどうかを決定します。以下の2つのオプションから選択できます。

* Launch エージェントに Docker イメージを構築させ、Amazon ECR にイメージをプッシュし、[SageMaker トレーニング](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html)ジョブを送信させます。このオプションは、MLエンジニアがトレーニングコードを迅速に反復するのに適しています。
* Launch エージェントが、トレーニングまたは推論スクリプトを含む既存の Docker イメージを使用します。このオプションは既存の CI システムと良く連携します。このオプションを選択した場合は、Docker イメージを手動で Amazon ECR にアップロードする必要があります。

### AWS リソースのセットアップ

以下の AWS リソースを、お好みの AWS リージョンに設定してください。

1. コンテナイメージを保存するための [ECR リポジトリ](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html)。
2. SageMaker トレーニングジョブの入力と出力を保存するための1つ以上の [S3 バケット](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)。
3. SageMaker トレーニングジョブを実行し、Amazon ECR や Amazon S3 と連携するための IAM ロール。

これらのリソースの ARN をメモしておいてください。Launch キュー設定を定義する際に ARN が必要になります。

### Launch エージェントのために IAM ロールを作成する

Launch エージェントは、Amazon SageMaker トレーニングジョブを作成するための権限が必要です。以下の手順に従って IAM ロールを作成してください。

1. AWS の IAM 画面から新規ロールを作成します。
2. **信頼されたエンティティ** には **AWS アカウント** (または組織のポリシーに適した他のオプション) を選択します。
3. 権限画面をスクロールして **次へ** をクリックします。
4. ロールに名前と説明を付けます。
5. **ロールの作成** を選択します。
6. **権限の追加** の下で **インラインポリシーの作成** を選択します。
7. JSON ポリシーエディタに切り替え、以下のポリシーをユースケースに基づいて貼り付けます。 `<>` で囲まれた値は自分の値に置き換えます:

<Tabs
  defaultValue="build"
  values={[
    {label: 'エージェントが Docker イメージを構築して提出', value: 'build'},
    {label: 'エージェントが事前構築された Docker イメージを提出', value: 'no-build'},
  ]}>
  <TabItem value="no-build">

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
        "Resource": "arn:aws:SageMaker:<region>:<account-id>:*"
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
  </TabItem>
  <TabItem value="build">

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
        "Resource": "arn:aws:SageMaker:<region>:<account-id>:*"
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
  </TabItem>
</Tabs>

8. **次へ** をクリックします。
9. ロールの ARN をメモしておきます。設定する際に必要になります。

IAM ロールの作成方法についての詳細は、[AWS Identity and Access Management Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html) を参照してください。

:::info
* Launch エージェントにイメージを構築させるには、[高度なエージェント設定](./setup-agent-advanced.md)で追加の権限が必要です。
* SageMaker キューの `kms:CreateGrant` 権限は、関連する ResourceConfig に VolumeKmsKeyId が指定されており、関連するロールにこのアクションを許可するポリシーがない場合にのみ必要です。
:::

## SageMaker 用に Launch キューを設定する

次に、SageMaker をコンピュートリソースとして使用するキューを W&B アプリで作成します:

1. [Launch アプリ](https://wandb.ai/launch) に移動します。
2. **Create Queue** ボタンをクリックします。
3. キューを作成する **エンティティ** を選択します。
4. **Name** フィールドにキューの名前を入力します。
6. **Resource** として **SageMaker** を選択します。
7. **Configuration** フィールドに SageMaker ジョブに関する情報を入力します。デフォルトでは、W&B は YAML と JSON の `CreateTrainingJob` リクエストボディを自動的に入力します:
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
最低限指定しなければならないのは以下の項目です:

- `RoleArn`: SageMaker 実行 IAM ロールの ARN（[必要条件](#prerequisites)参照）。Launch エージェント IAM ロールと混同しないでください。
- `OutputDataConfig.S3OutputPath`: SageMaker の出力を保存する Amazon S3 URI。
- `ResourceConfig`: リソース設定の必須仕様。リソース設定のオプションは[こちら](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_ResourceConfig.html)に記載されています。
- `StoppingCondition`: トレーニングジョブの停止条件の必須仕様。オプションは[こちら](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_StoppingCondition.html)に記載されています。
7. **Create Queue** ボタンをクリックします。

## Launch エージェントのセットアップ

以下のセクションでは、エージェントをどこにデプロイし、エージェントの設定方法について説明します。

Amazon SageMaker 用の W&B Launch キューには、エージェントのデプロイ方法が[いくつかのオプション](#decide-where-to-run-the-launch-agent)あります: ローカルマシン、EC2 インスタンス、または EKS クラスター。エージェントをデプロイする場所に基づいて[適切にエージェントを設定](#configure-a-launch-agent)します。

### Launch エージェントを実行する場所を決定する

プロダクションワークロードおよび既に EKS クラスターを持つ顧客向けには、W&B はこの Helm チャートを使用して EKS クラスターに Launch エージェントをデプロイすることを強く推奨します。

既存の EKS クラスターがない場合、プロダクションワークロードには EC2 インスタンスが良い選択肢です。Launch エージェントインスタンスは常時稼働しますが、エージェントには `t2.micro` サイズの EC2 インスタンス以上のものは必要ありません。

実験的またはソロのユースケースには、ローカルマシンで Launch エージェントを実行するのが迅速な方法です。

ユースケースに基づいて、Launch エージェントを適切に設定するために、以下のタブに従ってください:
<Tabs
  defaultValue="eks"
  values={[
    {label: 'EKS', value: 'eks'},
    {label: 'EC2', value: 'ec2'},
    {label: 'Local machine', value: 'local'},
  ]}>
  <TabItem value="eks">

W&B は、[W&B 管理の helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)を使用してエージェントを EKS クラスターにインストールすることを強く推奨します。

</TabItem>
  <TabItem value="ec2">

Amazon EC2 ダッシュボードに移動し、以下の手順を行います:

1. **Launch instance** をクリックします。
2. **Name** フィールドに名前を入力します。タグを追加することもできます。
2. **Instance type** から EC2 コンテナのインスタンスタイプを選択します。1vCPU と 1GiB のメモリ以上は必要ありません（例: t2.micro）。
3. **Key pair (login)** フィールドに組織のためのキーペアを作成します。このキーペアを使用して[SSH クライアントで EC2 インスタンスに接続](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html)します。
2. **Network settings** で組織に適したセキュリティグループを選択します。
3. **Advanced details** を展開します。**IAM instance profile** で上記で作成した Launch エージェント IAM ロールを選択します。
2. **Summary** フィールドを確認し、正しければ **Launch instance** を選択します。

AWS の EC2 ダッシュボードの左パネルで **Instances** に移動します。作成した EC2 インスタンスが実行中であることを確認します（**インスタンスステータス** 列を参照）。EC2 インスタンスが実行中であることを確認したら、ローカルマシンのターミナルに移動し、以下を行います:

1. **Connect** を選択します。
2. **SSH クライアント** タブを選択し、EC2 インスタンスに接続するための手順に従います。
3. EC2 インスタンス内で以下のパッケージをインストールします:
```bash
sudo yum install python311 -y && python3 -m ensurepip --upgrade && pip3 install wandb && pip3 install wandb[launch]
```
4. 次に、EC2 インスタンスに Docker をインストールし、起動します:
```bash
sudo yum update -y && sudo yum install -y docker python3 && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -a -G docker ec2-user

newgrp docker
```

これで Launch エージェントの設定を進めることができます。

  </TabItem>
  <TabItem value="local">

エージェントがローカルマシンでポーリングする際には、`~/.aws/config` および `~/.aws/credentials` にある AWS 設定ファイルを使用してエージェントにロールを関連付けます。前のステップで作成した Launch エージェント IAM ロールの ARN を提供します。

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

セッショントークンは、関連するプリンシパルに基づき [最大1時間または3日間有効](https://docs.aws.amazon.com/cli/latest/reference/sts/get-session-token.html#description)であることに注意してください。

  </TabItem>
</Tabs>

### Launch エージェントを設定する

エージェントを YAML 設定ファイル `launch-config.yaml` で設定します。

デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` に設定ファイルをチェックします。Launch エージェントを起動するときに `-c` フラグを使用して別のディレクトリを指定することもできます。

以下の YAML スニペットは、主要な設定エージェントオプションを指定する方法を示しています:

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

`wandb launch-agent` でエージェントを起動します。

## (Optional) Launch ジョブ Docker イメージを Amazon ECR にプッシュする

:::info
このセクションは、既存のトレーニングまたは推論ロジックを含む Docker イメージを使用する場合にのみ適用されます。[