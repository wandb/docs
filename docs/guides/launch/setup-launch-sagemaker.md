---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Set up for SageMaker

W&B Launch を使用して、Amazon SageMaker に launch ジョブを送信し、提供されたアルゴリズムやカスタムアルゴリズムを使用して SageMaker プラットフォーム上で機械学習モデルをトレーニングすることができます。SageMaker は計算リソースのスピンアップとリリースを管理するため、EKS クラスターを持たないチームにとって良い選択肢となります。

Amazon SageMaker に接続された W&B Launch キューに送信された Launch ジョブは、[CreateTrainingJob API](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_CreateTrainingJob.html) を使用して SageMaker トレーニングジョブとして実行されます。Launch キューの設定を使用して、`CreateTrainingJob` API に送信される引数を制御します。

Amazon SageMaker は、[Docker イメージを使用してトレーニングジョブを実行します](https://docs.aws.amazon.com/SageMaker/latest/dg/your-algorithms-training-algo-dockerfile.html)。SageMaker によってプルされるイメージは Amazon Elastic Container Registry (ECR) に保存されている必要があります。つまり、トレーニングに使用するイメージは ECR に保存する必要があります。

:::note
このガイドは、SageMaker トレーニングジョブを実行する方法を示しています。Amazon SageMaker での推論のためにモデルをデプロイする方法については、[このLaunchジョブの例](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints)をご覧ください。
:::

## Prerequisites

始める前に、次の前提条件を満たしていることを確認してください：

* [LaunchエージェントにDockerイメージを作成させるかどうかを決定する。](#decide-if-you-want-the-launch-agent-to-build-a-docker-images)
* [AWSリソースを設定し、S3、ECR、およびSagemaker IAMロールに関する情報を収集する。](#set-up-aws-resources)
* [LaunchエージェントのIAMロールを作成する。](#create-an-iam-role-for-launch-agent)

### Decide if you want the Launch agent to build Docker images

W&B Launch エージェントに Docker イメージを作成させるかどうかを決定します。選択肢は2つあります：

* LaunchエージェントにDockerイメージを作成させ、イメージをAmazon ECRにプッシュして、[SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html)ジョブを送信させる。このオプションは、トレーニングコードを迅速に繰り返し作成するMLエンジニアにとって簡素化を提供することができます。
* Launchエージェントがトレーニングまたは推論スクリプトを含む既存のDockerイメージを使用する。このオプションは既存のCIシステムとよく連携します。このオプションを選択する場合、DockerイメージをAmazon ECRのコンテナレジストリに手動でアップロードする必要があります。

### Set up AWS resources

お好みのAWSリージョンで次のAWSリソースを設定していることを確認してください：

1. コンテナイメージを保存するための[ECRリポジトリ](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html)。
2. SageMaker トレーニングジョブの入力と出力を保存するための[S3バケット](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)1つ以上。
3. Amazon SageMaker がトレーニングジョブを実行し、Amazon ECR および Amazon S3 と連携することを許可するための IAM ロール。

これらのリソースの ARN をメモしておいてください。 Launch キューの設定を定義するときに ARN が必要となります。

### Create an IAM role for Launch agent

LaunchエージェントにAmazon SageMakerトレーニングジョブを作成する権限が必要です。次の手順に従ってIAMロールを作成してください：

1. AWSのIAM画面から新しいロールを作成します。
2. **Trusted Entity** に **AWS Account** （または組織のポリシーに適した他のオプション）を選択します。
3. 権限画面をスクロールし、**Next**をクリックします。
4. ロールに名前と説明を付けます。
5. **Create role** を選択します。
6. **Add permissions** の下で、**Create inline policy** を選択します。
7. JSONポリシーエディタに切り替え、以下のユースケースに基づいたポリシーを貼り付けます。`<>`で囲まれた値を自分の値に置き換えます：

<Tabs
  defaultValue="build"
  values={[
    {label: 'Agent builds and submits Docker image', value: 'build'},
    {label: 'Agent submits pre-built Docker image', value: 'no-build'},
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

8. **Next** をクリックします。
9. ロールのARNをメモしておいてください。このARNはLaunchエージェントを設定するときに指定します。

IAMロールの作成方法について詳しくは、[AWS Identity and Access Management Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html) を参照してください。

:::info
* Launchエージェントにイメージを作成させる場合、追加の権限が必要となるので、[Advanced agent set up](./setup-agent-advanced.md) を参照してください。
* SageMakerキューに対する `kms:CreateGrant` 権限は、関連する ResourceConfig に指定された VolumeKmsKeyId があり、関連するロールにこのアクションを許可するポリシーがない場合にのみ必要です。
:::

## Configure launch queue for SageMaker

次に、W&Bアプリで SageMaker を計算リソースとして使用するキューを作成します：

1. [Launch App](https://wandb.ai/launch) に移動します。
2. **Create Queue** ボタンをクリックします。
3. **Entity** を選択します。
4. キューに名前を付けます。
5. **Resource** として **SageMaker** を選択します。
6. **Configuration** フィールドに SageMaker ジョブに関する情報を提供します。デフォルトでは、W&B が YAML および JSON の `CreateTrainingJob` リクエストボディを生成します：
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
少なくとも次を指定する必要があります：

- `RoleArn` : SageMaker 実行 IAM ロールの ARN（[prerequisites](#prerequisites) を参照）。Launch **agent** IAM ロールと混同しないでください。
- `OutputDataConfig.S3OutputPath` : SageMaker の出力が保存される Amazon S3 URI。
- `ResourceConfig`：リソース設定の必須仕様。リソース設定のオプションは[こちら](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_ResourceConfig.html)に記載。
- `StoppingCondition`：トレーニングジョブの停止条件の必須仕様。オプションは[こちら](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_StoppingCondition.html)に記載。

7. **Create Queue** ボタンをクリックします。

## Set up the launch agent

次のセクションでは、エージェントのデプロイ場所と、デプロイ場所に基づくエージェントの設定方法について説明します。

Amazon SageMaker の Launch キューは、[ローカルマシン、EC2インスタンス、EKSクラスターにデプロイするオプションがあります](#decide-where-to-run-the-launch-agent)。デプロイ先に応じて、[Launchエージェントを適切に設定してください](#configure-a-launch-agent)。

### Decide where to run the Launch agent

プロダクションワークロードや既に EKS クラスターを持つ顧客に対しては、このヘルムチャートを使用して Launch エージェントを EKS クラスターにデプロイすることを W&B は強く推奨します。

現在 EKS クラスターを持っていないプロダクションワークロードには、EC2インスタンスが良い選択肢です。launch-agentインスタンスは常に稼働していますが、t2.micro程度のEC2インスタンスで十分です。

実験的や個人のユースケースには、ローカルマシンでLaunchエージェントを稼働させることが高速で開始する方法です。

ユースケースに基づいて、以下のタブに記載された手順に従って、Launchエージェントを適切に設定してください：
<Tabs
  defaultValue="eks"
  values={[
    {label: 'EKS', value: 'eks'},
    {label: 'EC2', value: 'ec2'},
    {label: 'Local machine', value: 'local'},
  ]}>
  <TabItem value="eks">

W&Bは、エージェントをEKSクラスターにインストールするために[W&B 管理のヘルムチャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)を使用することを強く推奨します。

  </TabItem>
  <TabItem value="ec2">

Amazon EC2 ダッシュボードに移動し、次の手順を完了してください：

1. **Launch instance** をクリックします。
2. **Name** フィールドに名前を入力します。オプションでタグを追加します。
2. **Instance type** からお好みのインスタンスタイプを選択します。1vCPUと1GiBのメモリ（例：t2.micro）で十分です。
3. **Key pair (login)** フィールドで組織のためにキーペアを作成します。このキーペアは後でSSHクライアントを使用して[EC2インスタンスに接続する](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html)際に使用します。
2. **Network settings** で組織に適したセキュリティグループを選択します。
3. **Advanced details** を展開します。**IAM instance profile** で上記で作成したLaunchエージェントIAMロールを選択します。
2. **Summary** フィールドを確認します。正しければ **Launch instance** を選択します。

AWS の EC2 ダッシュボードの左側パネルで **Instances** に移動します。作成した EC2 インスタンスが稼働中であることを確認します（**Instance state** 列を参照）。EC2 インスタンスが稼働中であることを確認したら、ローカルマシンのターミナルに移動し、次の手順を完了します：

1. **Connect** を選択します。
2. **SSH client** タブを選択し、EC2インスタンスに接続するための指示に従ってください。
3. EC2 インスタンス内で次のパッケージをインストールします：
```bash
sudo yum install python311 -y && python3 -m ensurepip --upgrade && pip3 install wandb && pip3 install wandb[launch]
```
4. 次に、EC2 インスタンス内で Docker をインストールして起動します：
```bash
sudo yum update -y && sudo yum install -y docker python3 && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -a -G docker ec2-user

newgrp docker
```

これでLaunchエージェントの設定に進むことができます。

  </TabItem>
  <TabItem value="local">

ローカルマシンでポーリングするエージェントにロールを関連付けるために、`~/.aws/config` と `~/.aws/credentials` にある AWS 設定ファイルを使用します。前の手順で作成したLaunchエージェントのIAMロールARNを提供します。

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

セッショントークンには、関連するプリンシパルに応じて最大1時間または3日の[最大長](https://docs.aws.amazon.com/cli/latest/reference/sts/get-session-token.html#description)があることに注意してください。

  </TabItem>
</Tabs>

### Configure a launch agent

`launch-config.yaml` という名前の YAML 設定ファイルで Launch エージェントを設定します。

デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` にこの設定ファイルをチェックします。Launch エージェントを起動する際に `-c` フラグで別のディレクトリを指定することもできます。

次の YAML コードスニペットは、コアエージェントオプションの設定方法を示しています：

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

次に、`wandb launch-agent` を使用してエージェントを開始します。

## (Optional) Push your launch job Docker image to Amazon ECR

:::info
このセクションは、トレーニングまたは推論ロジックを含む既存のDockerイメージを使用するLaunchエージェントにのみ適用されます。[Launchエージェントの動作方法には2つのオプションがあります。](#decide-if-you-want-the-launch-agent-to-build-a-docker-images)
:::

Launch ジョブが含まれる Docker イメージを Amazon ECR レポジトリにアップロードします。イメージベースのジョブを使用している場合、Docker イメージは新しい Launch ジョブを送信する前に ECR レジストリに保存されている必要があります。