---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# SageMaker のセットアップ

W&B Launch を使用して、Amazon SageMaker にジョブを送信し、提供されたアルゴリズムやカスタムアルゴリズムを使用して SageMaker プラットフォーム上で機械学習モデルをトレーニングできます。SageMaker はコンピューティングリソースの起動と解放を管理するため、EKS クラスターを持たないチームにとって良い選択肢となります。

Amazon SageMaker に接続された W&B Launch キューに送信されたジョブは、[CreateTrainingJob API](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_CreateTrainingJob.html) を使用して SageMaker トレーニングジョブとして実行されます。Launch キューの設定を使用して、`CreateTrainingJob` API に送信される引数を制御します。

Amazon SageMaker は[トレーニングジョブを実行するために Docker イメージを使用します](https://docs.aws.amazon.com/SageMaker/latest/dg/your-algorithms-training-algo-dockerfile.html)。SageMaker によってプルされるイメージは Amazon Elastic Container Registry (ECR) に保存されている必要があります。つまり、トレーニングに使用するイメージは ECR に保存されている必要があります。

:::note
このガイドは SageMaker トレーニングジョブの実行方法を示しています。Amazon SageMaker で推論用のモデルをデプロイする方法については、[この例の Launch ジョブ](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints)を参照してください。
:::

## 前提条件

始める前に、以下の前提条件を満たしていることを確認してください：

* [Launch エージェントに Docker イメージをビルドさせるかどうかを決定します。](#decide-if-you-want-the-launch-agent-to-build-a-docker-images)
* [AWS リソースを設定し、S3、ECR、および SageMaker IAM ロールに関する情報を収集します。](#set-up-aws-resources)
* [Launch エージェント用の IAM ロールを作成します。](#create-an-iam-role-for-launch-agent)

### Launch エージェントに Docker イメージをビルドさせるかどうかを決定します

W&B Launch エージェントに Docker イメージをビルドさせるかどうかを決定します。選択肢は2つあります：

* Launch エージェントに Docker イメージをビルドさせ、イメージを Amazon ECR にプッシュし、[SageMaker トレーニング](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html)ジョブを送信させる。このオプションは、トレーニングコードを迅速に反復する ML エンジニアにとって簡便です。
* トレーニングまたは推論スクリプトを含む既存の Docker イメージを使用する。このオプションは既存の CI システムとよく連携します。このオプションを選択する場合、Docker イメージを手動で Amazon ECR のコンテナレジストリにアップロードする必要があります。

### AWS リソースの設定

希望する AWS リージョンに以下の AWS リソースが設定されていることを確認してください：

1. コンテナイメージを保存するための [ECR リポジトリ](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html)。
2. SageMaker トレーニングジョブの入力と出力を保存するための1つ以上の [S3 バケット](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)。
3. SageMaker がトレーニングジョブを実行し、Amazon ECR および Amazon S3 と連携することを許可する IAM ロール。

これらのリソースの ARN をメモしておいてください。Launch キューの設定を定義する際に ARN が必要になります。

### Launch エージェント用の IAM ロールを作成します

Launch エージェントが Amazon SageMaker トレーニングジョブを作成するための権限が必要です。以下の手順に従って IAM ロールを作成します：

1. AWS の IAM 画面から新しいロールを作成します。
2. **Trusted Entity** には **AWS Account**（または組織のポリシーに適した他のオプション）を選択します。
3. 権限画面をスクロールして **Next** をクリックします。
4. ロールに名前と説明を付けます。
5. **Create role** を選択します。
6. **Add permissions** の下で **Create inline policy** を選択します。
7. JSON ポリシーエディタに切り替え、以下のポリシーをユースケースに基づいて貼り付けます。`<>` で囲まれた値を自分の値に置き換えます：

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
9. ロールの ARN をメモします。Launch エージェントを設定する際に ARN を指定します。

IAM ロールの作成方法について詳しくは、[AWS Identity and Access Management Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html) を参照してください。

:::info
* Launch エージェントにイメージをビルドさせる場合は、追加の権限が必要なため、[Advanced agent set up](./setup-agent-advanced.md) を参照してください。
* SageMaker キューの `kms:CreateGrant` 権限は、関連する ResourceConfig に指定された VolumeKmsKeyId があり、関連するロールにこのアクションを許可するポリシーがない場合にのみ必要です。
:::

## SageMaker 用の Launch キューを設定する

次に、W&B アプリで SageMaker をコンピューティングリソースとして使用するキューを作成します：

1. [Launch アプリ](https://wandb.ai/launch)に移動します。
3. **Create Queue** ボタンをクリックします。
4. キューを作成する **Entity** を選択します。
5. **Name** フィールドにキューの名前を入力します。
6. **Resource** として **SageMaker** を選択します。
7. **Configuration** フィールドに SageMaker ジョブに関する情報を提供します。デフォルトでは、W&B は YAML および JSON の `CreateTrainingJob` リクエストボディを入力します：
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
最低限、以下を指定する必要があります：

- `RoleArn` : SageMaker 実行 IAM ロールの ARN（[前提条件](#prerequisites)を参照）。Launch エージェント IAM ロールと混同しないでください。
- `OutputDataConfig.S3OutputPath` : SageMaker の出力が保存される Amazon S3 URI。
- `ResourceConfig`: リソース設定の必須仕様。リソース設定のオプションは[こちら](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_ResourceConfig.html)に記載されています。
- `StoppingCondition`: トレーニングジョブの停止条件の必須仕様。オプションは[こちら](https://docs.aws.amazon.com/SageMaker/latest/APIReference/API_StoppingCondition.html)に記載されています。
7. **Create Queue** ボタンをクリックします。

## Launch エージェントのセットアップ

次のセクションでは、エージェントをデプロイする場所と、デプロイ先に基づいてエージェントを設定する方法について説明します。

Amazon SageMaker 用の Launch エージェントは、ローカルマシン、EC2 インスタンス、または EKS クラスターでデプロイする[いくつかのオプション](#decide-where-to-run-the-launch-agent)があります。エージェントをデプロイする場所に基づいて、適切に Launch エージェントを設定します。

### Launch エージェントを実行する場所を決定する

プロダクションワークロードおよび既に EKS クラスターを持っている顧客には、Helm チャートを使用して Launch エージェントを EKS クラスターにデプロイすることをお勧めします。

現在 EKS クラスターを持っていないプロダクションワークロードには、EC2 インスタンスが良いオプションです。Launch エージェントインスタンスは常に稼働し続けますが、エージェントには `t2.micro` サイズの EC2 インスタンスで十分です。

実験的または個人用のユースケースには、ローカルマシンで Launch エージェントを実行することが迅速なスタート方法となります。

ユースケースに基づいて、以下のタブで提供される指示に従って Launch エージェントを適切に設定してください：
<Tabs
  defaultValue="eks"
  values={[
    {label: 'EKS', value: 'eks'},
    {label: 'EC2', value: 'ec2'},
    {label: 'Local machine', value: 'local'},
  ]}>
  <TabItem value="eks">

W&B は、エージェントを EKS クラスターにインストールするために[W&B 管理の Helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)を使用することを強く推奨します。

</TabItem>
  <TabItem value="ec2">

Amazon EC2 ダッシュボードに移動し、以下の手順を完了します：

1. **Launch instance** をクリックします。
2. **Name** フィールドに名前を入力します。オプションでタグを追加します。
2. **Instance type** から EC2 コンテナのインスタンスタイプを選択します。1vCPU と 1GiB のメモリ（例：t2.micro）で十分です。
3. **Key pair (login)** フィールドで組織用のキーペアを作成します。このキーペアを使用して、後のステップで SSH クライアントを使用して [EC2 インスタンスに接続](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html)します。
2. **Network settings** で、組織に適したセキュリティグループを選択します。
3. **Advanced details** を展開します。**IAM instance profile** には、上記で作成した Launch エージェント IAM ロールを選択します。
2. **Summary** フィールドを確認します。正しければ、**Launch instance** を選択します。

AWS の EC2 ダッシュボードの左パネルで **Instances** に移動します。作成した EC2 インスタンスが稼働していることを確認します（**Instance state** 列を参照）。EC2 インスタンスが稼働していることを確認したら、ローカルマシンの端末に移動し、以下を完了します：

1. **Connect** を選択します。
2. **SSH client** タブを選択し、EC2 インスタンスに接続するための指示に従います。
3. EC2 インスタンス内で以下のパッケージをインストールします：
```bash
sudo yum install python311 -y && python3 -m ensurepip --upgrade && pip3 install wandb && pip3 install wandb[launch]
```
4. 次に、EC2 インスタンス内で Docker をインストールして起動します：
```bash
sudo yum update -y && sudo yum install -y docker python3 && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -a -G docker ec2-user

newgrp docker
```

これで Launch エージェントの設定を行う準備が整いました。

  </TabItem>
  <TabItem value="local">

ローカルマシンでポーリングしているエージェントにロールを関連付けるために、`~/.aws/config` および `~/.aws/credentials` にある AWS 設定ファイルを使用します。前のステップで作成した Launch エージェント用の IAM ロール ARN を提供します。

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

セッショントークンの有効期限は、関連するプリンシパルに応じて[最大1時間または3日](https://docs.aws.amazon.com/cli/latest/reference/sts/get-session-token.html#description)です。

  </TabItem>
</Tabs>

### Launch エージェントの設定
Launch エージェントを `launch-config.yaml` という名前の YAML 設定ファイルで設定します。

デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` に設定ファイルをチェックします。Launch エージェントを起動する際に `-c` フラグを使用して、別のディレクトリを指定することもできます。

以下の YAML スニペットは、コア設定エージェントオプションを指定する方法を示しています：

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

次に、`wandb launch-agent` でエージェントを起動します。

## （オプション）Launch ジョブの Docker イメージを Amazon ECR にプッシュする

:::info
このセクションは、トレーニングまたは推論ロジックを含む既存の Docker イメージを使用する Launch エージェントにのみ適用されます。[Launch エージェントの動作方法には2つのオプ