---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Set up for SageMaker

W&B Launchを使用して、Amazon SageMakerにlaunchジョブを送信し、提供されたアルゴリズムやカスタムアルゴリズムを使用してSageMakerプラットフォーム上で機械学習モデルをトレーニングすることができます。SageMakerはコンピュートリソースの起動と解放を管理するため、EKSクラスターを持たないチームにとって良い選択肢となります。

Amazon SageMakerに接続されたW&B Launchキューに送信されたlaunchジョブは、[CreateTrainingJob API](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_CreateTrainingJob.html)を使用してSageMakerトレーニングジョブとして実行されます。launchキュー設定を使用して、`CreateTrainingJob` APIに送信される引数を制御します。

Amazon SageMakerは[Dockerイメージを使用してトレーニングジョブを実行します](https://docs.aws.amazon.com/SageMaker/latest/dg/your-algorithms-training-algo-dockerfile.html)。SageMakerによってプルされるイメージはAmazon Elastic Container Registry (ECR)に保存されている必要があります。つまり、トレーニングに使用するイメージはECRに保存されている必要があります。

:::note
このガイドはSageMakerトレーニングジョブの実行方法を示しています。Amazon SageMakerで推論用のモデルをデプロイする方法については、[この例のLaunchジョブ](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints)を参照してください。
:::

## Prerequisites

始める前に、以下の前提条件を満たしていることを確認してください：

* [LaunchエージェントにDockerイメージをビルドさせるかどうかを決定します。](#decide-if-you-want-the-launch-agent-to-build-a-docker-images)
* [AWSリソースを設定し、S3、ECR、およびSageMaker IAMロールに関する情報を収集します。](#set-up-aws-resources)
* [Launchエージェント用のIAMロールを作成します。](#create-an-iam-role-for-launch-agent)

### Decide if you want the Launch agent to build a Docker images

W&B LaunchエージェントにDockerイメージをビルドさせるかどうかを決定します。選択肢は2つあります：

* LaunchエージェントにDockerイメージをビルドさせ、イメージをAmazon ECRにプッシュし、[SageMakerトレーニング](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html)ジョブを送信させる。このオプションは、トレーニングコードを迅速に繰り返すMLエンジニアにとって簡便です。
* 既存のDockerイメージを使用し、その中にトレーニングまたは推論スクリプトを含める。このオプションは既存のCIシステムとよく連携します。このオプションを選択する場合、Dockerイメージを手動でAmazon ECRのコンテナレジストリにアップロードする必要があります。

### Set up AWS resources

以下のAWSリソースが希望するAWSリージョンに設定されていることを確認してください：

1. コンテナイメージを保存するための[ECRリポジトリ](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html)。
2. SageMakerトレーニングジョブの入力と出力を保存するための1つ以上の[S3バケット](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)。
3. SageMakerがトレーニングジョブを実行し、Amazon ECRおよびAmazon S3と連携することを許可するIAMロール。

これらのリソースのARNをメモしておいてください。Launchキュー設定を定義するときにARNが必要になります。

### Create an IAM role for Launch agent

LaunchエージェントがAmazon SageMakerトレーニングジョブを作成するための権限が必要です。以下の手順に従ってIAMロールを作成します：

1. AWSのIAM画面から新しいロールを作成します。
2. **Trusted Entity**には**AWS Account**（または組織のポリシーに適した他のオプション）を選択します。
3. 権限画面をスクロールして**Next**をクリックします。
4. ロールに名前と説明を付けます。
5. **Create role**を選択します。
6. **Add permissions**の下で**Create inline policy**を選択します。
7. JSONポリシーエディタに切り替え、以下のポリシーをユースケースに基づいて貼り付けます。`<>`で囲まれた値を自分の値に置き換えます：

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

8. **Next**をクリックします。
9. ロールのARNをメモします。launchエージェントを設定する際にARNを指定します。

IAMロールの作成方法について詳しくは、[AWS Identity and Access Management Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html)を参照してください。

:::info
* Launchエージェントにイメージをビルドさせる場合、追加の権限が必要なため、[Advanced agent set up](./setup-agent-advanced.md)を参照してください。
* SageMakerキューの`kms:CreateGrant`権限は、関連するResourceConfigに指定されたVolumeKmsKeyIdがあり、関連するロールにこのアクションを許可するポリシーがない場合にのみ必要です。
:::

## Configure launch queue for SageMaker

次に、W&BアプリでSageMakerをコンピュートリソースとして使用するキューを作成します：

1. [Launch App](https://wandb.ai/launch)に移動します。
3. **Create Queue**ボタンをクリックします。
4. キューを作成したい**Entity**を選択します。
5. **Name**フィールドにキューの名前を入力します。
6. **Resource**として**SageMaker**を選択します。
7. **Configuration**フィールドにSageMakerジョブに関する情報を入力します。デフォルトで、W&BはYAMLおよびJSONの`CreateTrainingJob`リクエストボディを入力します：
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
最低限指定する必要があるのは：

- `RoleArn` : SageMaker実行IAMロールのARN（[前提条件](#prerequisites)を参照）。launchエージェントIAMロールと混同しないでください。
- `OutputDataConfig.S3OutputPath` : SageMakerの出力が保存されるAmazon S3 URI。
- `ResourceConfig`: リソース設定の必須仕様。リソース設定のオプションは[こちら](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_ResourceConfig.html)に記載されています。
- `StoppingCondition`: トレーニングジョブの停止条件の必須仕様。オプションは[こちら](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_StoppingCondition.html)に記載されています。
7. **Create Queue**ボタンをクリックします。

## Set up the launch agent

次のセクションでは、エージェントをデプロイする場所と、デプロイ場所に基づいてエージェントを設定する方法について説明します。

Amazon SageMakerのためのlaunchキューには、エージェントをデプロイする[いくつかのオプション](#decide-where-to-run-the-launch-agent)があります：ローカルマシン、EC2インスタンス、またはEKSクラスター。エージェントをデプロイする場所に基づいて、[適切にlaunchエージェントを設定](#configure-a-launch-agent)します。

### Decide where to run the Launch agent

プロダクションワークロードおよび既にEKSクラスターを持っている顧客には、Helmチャートを使用してLaunchエージェントをEKSクラスターにデプロイすることをお勧めします。

現在EKSクラスターを持たないプロダクションワークロードには、EC2インスタンスが良いオプションです。launchエージェントインスタンスは常に稼働していますが、エージェントには`t2.micro`サイズのEC2インスタンスで十分です。

実験的または個人使用の場合、ローカルマシンでLaunchエージェントを実行するのが迅速な方法です。

ユースケースに基づいて、以下のタブに記載された手順に従ってlaunchエージェントを適切に設定してください：
<Tabs
  defaultValue="eks"
  values={[
    {label: 'EKS', value: 'eks'},
    {label: 'EC2', value: 'ec2'},
    {label: 'Local machine', value: 'local'},
  ]}>
  <TabItem value="eks">

W&Bは、EKSクラスターにエージェントをインストールするために[W&B管理のHelmチャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)を使用することを強く推奨します。

</TabItem>
  <TabItem value="ec2">

Amazon EC2ダッシュボードに移動し、以下の手順を完了します：

1. **Launch instance**をクリックします。
2. **Name**フィールドに名前を入力します。オプションでタグを追加します。
2. **Instance type**からEC2コンテナのインスタンスタイプを選択します。1vCPUと1GiBのメモリで十分です（例：t2.micro）。
3. **Key pair (login)**フィールドで組織用のキーペアを作成します。このキーペアを使用して、後のステップでSSHクライアントを使用して[EC2インスタンスに接続](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html)します。
2. **Network settings**で組織に適したセキュリティグループを選択します。
3. **Advanced details**を展開します。**IAM instance profile**には、上記で作成したlaunchエージェントIAMロールを選択します。
2. **Summary**フィールドを確認します。正しければ、**Launch instance**を選択します。

AWSのEC2ダッシュボードの左パネルで**Instances**に移動します。作成したEC2インスタンスが稼働していることを確認します（**Instance state**列を参照）。EC2インスタンスが稼働していることを確認したら、ローカルマシンの端末に移動し、以下を完了します：

1. **Connect**を選択します。
2. **SSH client**タブを選択し、EC2インスタンスに接続するための指示に従います。
3. EC2インスタンス内で以下のパッケージをインストールします：
```bash
sudo yum install python311 -y && python3 -m ensurepip --upgrade && pip3 install wandb && pip3 install wandb[launch]
```
4. 次に、EC2インスタンス内でDockerをインストールして起動します：
```bash
sudo yum update -y && sudo yum install -y docker python3 && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -a -G docker ec2-user

newgrp docker
```

これでLaunchエージェントの設定を行う準備が整いました。

  </TabItem>
  <TabItem value="local">

ローカルマシンでポーリングしているエージェントにロールを関連付けるために、`~/.aws/config`および`~/.aws/credentials`にあるAWS設定ファイルを使用します。前のステップで作成したlaunchエージェント用のIAMロールARNを提供します。

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

セッショントークンは、関連するプリンシパルに応じて、1時間または3日の[最大長](https://docs.aws.amazon.com/cli/latest/reference/sts/get-session-token.html#description)があります。

  </TabItem>
</Tabs>

### Configure a launch agent

launchエージェントを`launch-config.yaml`という名前のYAML設定ファイルで設定します。

デフォルトでは、W&Bは`~/.config/wandb/launch-config.yaml`に設定ファイルをチェックします。launchエージェントを起動する際に`-c`フラグを使用して、別のディレクトリを指定することもできます。

以下のYAMLスニペットは、コア設定エージェントオプションを指定する方法を示しています：

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

次に、`wandb launch-agent`でエージェントを起動します。

## (Optional) Push your launch job Docker image to Amazon ECR

:::info
このセクションは、トレーニングまたは推論ロジックを含む既存のDockerイメージを使用する場合にのみ適用されます。[Launchエージェントの動作方法には2つのオプションがあります。](#decide-if-you-want-the-launch-agent-to-build-a-docker-images)
:::

Launchジョブを含むDockerイメージをAmazon ECRリポジトリにアップロードします。イメージベースのジョブを使用する場合、新しいlaunchジョブを送信する前にDocker