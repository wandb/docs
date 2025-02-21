---
title: 'Tutorial: Set up W&B Launch on SageMaker'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-sagemaker
    parent: set-up-launch
url: guides/launch/setup-launch-sagemaker
---

W&B Launch を使用して、SageMaker プラットフォーム上で提供されたまたはカスタムのアルゴリズムを使用して機械学習モデルをトレーニングするためのローンチジョブを Amazon SageMaker に提出できます。SageMaker はコンピュートリソースのスピンアップとリリースを処理するため、EKS クラスターを持たないチームにとって良い選択肢となることがあります。

Amazon SageMaker に接続された W&B Launch キューに送信されたローンチジョブは、[CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) を使用して SageMaker トレーニングジョブとして実行されます。`CreateTrainingJob` API に送信される引数を制御するために、ローンチキュー設定を使用してください。

Amazon SageMaker は、[Docker イメージを使用してトレーニングジョブを実行](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html)します。SageMaker によってプルされたイメージは、Amazon Elastic Container Registry (ECR) に保存されている必要があります。これは、トレーニングに使用するイメージを ECR に保存しなければならないことを意味します。

{{% alert %}}
このガイドでは SageMaker トレーニングジョブを実行する方法を示しています。Amazon SageMaker での推論のためのモデルのデプロイについては、[この例のローンチジョブ](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints)を参照してください。
{{% /alert %}}

## 前提条件

始める前に、次の前提条件を満たしていることを確認してください。

* [Launch エージェントに Docker イメージを構築させるかどうかを決定してください。]({{< relref path="#decide-if-you-want-the-launch-agent-to-build-a-docker-images" lang="ja" >}})
* [AWS リソースを設定し、S3、ECR、SageMaker IAM ロールについての情報を収集してください。]({{< relref path="#set-up-aws-resources" lang="ja" >}})
* [Launch エージェント用の IAM ロールを作成してください。]({{< relref path="#create-an-iam-role-for-launch-agent" lang="ja" >}})

### Launch エージェントに Docker イメージを構築させるかどうかを決定

W&B の Launch エージェントに Docker イメージを構築させるかどうかを決定してください。選択できるオプションは次の 2 つです：

* Launch エージェントに Docker イメージの構築を許可し、Amazon ECR にイメージをプッシュして、[SageMaker トレーニング](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html)ジョブを提出させます。このオプションは、トレーニングコードを迅速に反復する ML エンジニアに一部のシンプルさを提供できます。
* Launch エージェントが、トレーニングまたは推論のスクリプトを含む既存の Docker イメージを使用します。このオプションは、既存の CI システムとよく連携します。このオプションを選択した場合、Amazon ECR のコンテナレジストリに Docker イメージを手動でアップロードする必要があります。

### AWS リソースを設定

以下の AWS リソースが、希望する AWS リージョンに設定されていることを確認してください：

1. コンテナイメージを保存するための [ECR リポジトリ](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html)。
2. SageMaker トレーニングジョブの入力と出力を保存するための、1 つ以上の [S3 バケット](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)。
3. SageMaker にトレーニングジョブを実行し、Amazon ECR および Amazon S3 と対話することを許可する Amazon SageMaker 用の IAM ロール。

これらのリソースの ARN をメモしてください。これらの ARN は、[Launch キュー設定]({{< relref path="#configure-launch-queue-for-sagemaker" lang="ja" >}}) を定義するときに必要になります。

### Launch エージェント用 IAM ポリシーを作成

1. AWS の IAM 画面から、新しいポリシーを作成します。
2. JSON ポリシーエディターに切り替え、ユースケースに基づいて次のポリシーを貼り付けます。`<>`で囲まれた値を、あなた自身の値に置き換えてください。

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

### Launch エージェント用に IAM ロールを作成

Launch エージェントには、Amazon SageMaker トレーニングジョブを作成する権限が必要です。以下の手順に従って IAM ロールを作成してください：

1. AWS の IAM 画面から、新しいロールを作成します。
2. **Trusted Entity** には **AWS アカウント**（または組織のポリシーに適した別のオプション）を選択します。
3. 権限画面をスクロールして、上記で作成したポリシー名を選択します。
4. ロールに名前と説明を付けます。
5. **Create role** を選択します。
6. ロールの ARN をメモしてください。Launch エージェントをセットアップするときに ARN を指定します。

IAM ロールの作成方法についての詳細は、[AWS Identity and Access Management Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html)を参照してください。

{{% alert %}}
* Launch エージェントにイメージを構築させたい場合は、追加の権限が必要な [Advanced agent set up]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) を参照してください。
* SageMaker キューの `kms:CreateGrant` 権限は、関連する ResourceConfig に指定された VolumeKmsKeyId がある場合、かつ関連するロールにこのアクションを許可するポリシーがない場合にのみ必要です。
{{% /alert %}}

## SageMaker 用のローンチキューを設定

次に、SageMaker を計算リソースとして使用する W&B アプリ内でキューを作成します：

1. [Launch アプリ](https://wandb.ai/launch)に移動します。
3. **Create Queue** ボタンをクリックします。
4. キューを作成したい **Entity** を選択します。
5. **Name** フィールドにキューの名前を入力します。
6. **Resource** として **SageMaker** を選択します。
7. **Configuration** フィールド内に SageMaker ジョブに関する情報を入力します。デフォルトで、W&B は YAML および JSON の `CreateTrainingJob` リクエストボディを入力します：
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
最低限指定する必要のあるものは次の通りです：

- `RoleArn` : SageMaker 実行 IAM ロールの ARN（[前提条件]({{< relref path="#prerequisites" lang="ja" >}})を参照）。Launch **エージェント** IAM ロールと混同しないでください。
- `OutputDataConfig.S3OutputPath` : SageMaker の出力が保存される Amazon S3 URI。
- `ResourceConfig`: 必須のリソース設定の具体化。リソース設定のオプションは[こちら](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceConfig.html)に記載されています。
- `StoppingCondition`: トレーニングジョブの停止条件の具体化が必要です。オプションは[ここ](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)に記載されています。
7. **Create Queue** ボタンをクリックします。

## Launch エージェントを設定

次のセクションは、エージェントをどこにデプロイするか、およびエージェントがデプロイされる場所に基づいてエージェントを設定する方法を説明しています。

[Amazon SageMaker 用のローンチエージェントがデプロされるためのオプションは複数あります]({{< relref path="#decide-where-to-run-the-launch-agent" lang="ja" >}}): ローカルマシン、EC2 インスタンス、または EKS クラスターに。エージェントをデプロイする場所に基づいて[エージェントを適切に設定します]({{< relref path="#configure-a-launch-agent" lang="ja" >}})。

### Launch エージェントを実行する場所を決定

プロダクション ワークロードと既に EKS クラスターを持っている顧客のために、W&B は Helm チャートを使用して EKS クラスターに Launch エージェントをデプロイすることを推奨しています。

現行の EKS クラスターがない場合は、EC2 インスタンスが良いオプションです。Launch エージェントインスタンスは常に稼働し続けますが、エージェントには相対的にコストが安価な `t2.micro` サイズの EC2 インスタンスが必要です。

実験的または単独のユースケースの場合、ローカルマシンで Launch エージェントを実行するのは、開始するための素早い方法です。

ユースケースに基づいて、以下のタブで提供されている手順に従って、Launch エージェントを適切に設定してください：
{{< tabpane text=true >}}
{{% tab "EKS" %}}
W&B はEKSクラスターでのエージェントのインストールに[W&B管理ヘルムチャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)の使用を強く推奨しています。
{{% /tab %}}
{{% tab "EC2" %}}
Amazon EC2 ダッシュボードに移動し、以下の手順を完了してください：

1. **Launch instance** をクリックします。
2. **Name** フィールドに名前を入力します。タグを追加することもできます。
2. **Instance type** から、EC2 コンテナのインスタンスタイプを選択します。1vCPU および 1GiB のメモリを超える必要はありません（たとえば、t2.micro）。
3. 組織内で、**Key pair (login)** フィールドにキーペアを作成します。後のステップで SSH クライアントを使用して [EC2 インスタンスに接続](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html)するために、このキーペアを使用します。
2. **Network settings** 内で、組織に適したセキュリティグループを選択します。
3. **Advanced details** を展開します。**IAM instance profile** には、上記で作成した Launch エージェント IAM ロールを選択します。
2. **Summary** フィールドを確認し、正しければ **Launch instance** を選択します。

AWS の EC2 ダッシュボードの左側のパネルで **Instances** に移動します。作成した EC2 インスタンスが実行されていることを確認してください（**Instance state** 列を参照）。EC2 インスタンスが実行中であることを確認したら、ローカルマシンのターミナルに移動し、以下を完了してください：

1. **Connect** を選択します。
2. **SSH client** タブを選び、EC2 インスタンスに接続するための手順に従います。
3. EC2 インスタンス内で、以下のパッケージをインストールします：
```bash
sudo yum install python311 -y && python3 -m ensurepip --upgrade && pip3 install wandb && pip3 install wandb[launch]
```
4. 次に、EC2 インスタンス内で Docker をインストールして起動します：
```bash
sudo yum update -y && sudo yum install -y docker python3 && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -a -G docker ec2-user

newgrp docker
```

これで Launch エージェントの設定を進めることができます。

{{% /tab %}}
{{% tab "Local machine" %}}

ローカルマシンでポーリングするエージェントに役割を関連付けるために、`~/.aws/config` および `~/.aws/credentials` にある AWS 設定ファイルを使用します。Launch エージェント用に作成した IAM ロール ARN を提供してください。
 
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

セッショントークンには[最大長](https://docs.aws.amazon.com/cli/latest/reference/sts/get-session-token.html#description)があり、それらが関連付けられているプリンシパルにより 1 時間または 3 日です。
{{% /tab %}}
{{< /tabpane >}}

### Launch エージェントを設定
YAML 設定ファイル `launch-config.yaml` を使用して Launch エージェントを設定します。

デフォルトでは、W&B は設定ファイルを `~/.config/wandb/launch-config.yaml` でチェックします。Launch エージェントを起動する際に `-c` フラグを使用して、別のディレクトリを指定することも可能です。

以下の YAML スニペットは、コア設定エージェントオプションを指定する方法を示します：

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

これで `wandb launch-agent` コマンドでエージェントを起動します。

## （オプション）Launch ジョブの Docker イメージを Amazon ECR にプッシュ

{{% alert %}}
このセクションは、Launch エージェントがトレーニングまたは推論ロジックを含む既存の Docker イメージを使用する場合のみに適用されます。[Launch エージェントがどのように動作するかに関するオプションは 2 つあります。]({{< relref path="#decide-if-you-want-the-launch-agent-to-build-a-docker-images" lang="ja" >}})  
{{% /alert %}}

Launch ジョブを含む Docker イメージを Amazon ECR レポジトリにアップロードします。Docker イメージを使用したジョブを使用している場合、ローンチジョブを新しく送信する前に、Docker イメージが ECR レジストリにある必要があります。