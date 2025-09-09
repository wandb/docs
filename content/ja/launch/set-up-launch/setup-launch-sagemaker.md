---
title: 'チュートリアル: SageMaker で W&B Launch をセットアップする'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-sagemaker
    parent: set-up-launch
url: guides/launch/setup-launch-sagemaker
---

W&B Launch を使うと、提供済みまたはカスタムのアルゴリズムで Amazon SageMaker 上で 機械学習 モデルのトレーニングを行うための Launch ジョブを送信できます。SageMaker はコンピュートリソースの起動と解放を担うため、EKS クラスターがない Teams にも適した選択肢です。

Amazon SageMaker に接続された W&B Launch キューに送られた Launch ジョブは、[CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) を用いる SageMaker の Training Job として実行されます。Launch キューの設定で、`CreateTrainingJob` API に渡す 引数 を制御できます。

Amazon SageMaker は [Docker イメージを使ってトレーニングジョブを実行](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html) します。SageMaker が pull するイメージは Amazon Elastic Container Registry (ECR) に保管されている必要があります。つまり、トレーニングに使うイメージは ECR に保管されていなければなりません。

{{% alert %}}
このガイドは SageMaker の Training Job の実行方法を説明します。Amazon SageMaker への推論用デプロイ方法は、[this example Launch job](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints) を参照してください。
{{% /alert %}}


## 前提条件

開始する前に、以下の前提条件を満たしていることを確認してください。

* [Launch エージェントに Docker イメージのビルドを任せるかどうかを決める]({{< relref path="#decide-if-you-want-the-launch-agent-to-build-a-docker-images" lang="ja" >}})
* [AWS リソースをセットアップし、S3、ECR、SageMaker の IAM ロールに関する情報を収集する]({{< relref path="#set-up-aws-resources" lang="ja" >}})
* [Launch エージェント用の IAM ロールを作成する]({{< relref path="#create-an-iam-role-for-launch-agent" lang="ja" >}})

### Launch エージェントに Docker イメージのビルドを任せるかどうかを決める

W&B Launch エージェントに Docker イメージのビルドを任せるかどうかを決めます。選択肢は次の 2 つです。

* Launch エージェントに Docker イメージのビルド、Amazon ECR へのプッシュ、[SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) ジョブの送信まで任せる。この選択肢は、トレーニング コードを素早く反復したい ML エンジニアにとってシンプルです。  
* 既存の Docker イメージ（トレーニングまたは推論のスクリプトを含む）を Launch エージェントで使用する。この選択肢は既存の CI システムとの相性が良いです。この場合、Docker イメージを手動で Amazon ECR のコンテナレジストリにアップロードする必要があります。


### AWS リソースをセットアップする

希望する AWS リージョンで、以下の AWS リソースが設定済みであることを確認してください。

1. コンテナイメージを保存するための [ECR リポジトリ](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html)。
2. SageMaker のトレーニングジョブの入力と出力を保存するための 1 つ以上の [S3 バケット](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)。
3. SageMaker がトレーニングジョブを実行し、Amazon ECR と Amazon S3 とやり取りできるようにするための Amazon SageMaker 用 IAM ロール。

これらリソースの ARN を控えておいてください。[Launch キュー設定]({{< relref path="#configure-launch-queue-for-sagemaker" lang="ja" >}}) を定義する際に必要になります。

### Launch エージェント用の IAM ポリシーを作成する

1. AWS の IAM 画面から、新しいポリシーを作成します。
2. JSON ポリシー エディタに切り替え、ユースケースに合わせて以下のポリシーを貼り付けます。`<>` で囲まれた 値 はご自身の 値 に置き換えてください。

{{< tabpane text=true >}}
{{% tab "エージェントが事前ビルド済み Docker イメージを送信" %}}
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
{{% tab "エージェントが Docker イメージをビルドして送信" %}}
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


### Launch エージェント用の IAM ロールを作成する

Launch エージェントには Amazon SageMaker のトレーニングジョブを作成する権限が必要です。以下の手順で IAM ロールを作成します。

1. AWS の IAM 画面から、新しいロールを作成します。 
2. **Trusted Entity** で **AWS Account**（または組織のポリシーに合う別のオプション）を選択します。
3. 権限の画面をスクロールして、上で作成したポリシー名を選択します。
4. ロールに名前と説明を付けます。
5. **Create role** を選択します。
6. ロールの ARN を控えます。後で Launch エージェントの設定時に ARN を指定します。

IAM ロールの作成方法は、[AWS Identity and Access Management Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html) を参照してください。

{{% alert %}}
* エージェントにイメージのビルドも任せたい場合は、追加で必要な権限について [Advanced agent set up]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) を参照してください。
* SageMaker キューに対する `kms:CreateGrant` 権限は、関連する ResourceConfig に VolumeKmsKeyId が指定されており、かつ関連ロールにその操作を許可するポリシーがない場合にのみ必要です。
{{% /alert %}}



## SageMaker 用に Launch キューを設定する

次に、W&B App でコンピュートリソースとして SageMaker を使うキューを作成します。

1. [Launch App](https://wandb.ai/launch) に移動します。
3. **Create Queue** ボタンをクリックします。
4. キューを作成したい **Entity** を選択します。
5. **Name** フィールドにキュー名を入力します。
6. **Resource** として **SageMaker** を選択します。
7. **Configuration** フィールド内で、SageMaker ジョブに関する情報を入力します。デフォルトでは、W&B が YAML と JSON の `CreateTrainingJob` リクエストボディを自動入力します:
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
最低限、以下を指定する必要があります:

- `RoleArn` : SageMaker 実行用 IAM ロールの ARN（[prerequisites]({{< relref path="#prerequisites" lang="ja" >}}) を参照）。Launch の **エージェント** IAM ロールと混同しないでください。
- `OutputDataConfig.S3OutputPath` : SageMaker の出力先を指定する Amazon S3 URI。
- `ResourceConfig`: リソース設定の必須指定。リソース設定のオプションは[こちら](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceConfig.html)。
- `StoppingCondition`: トレーニングジョブの停止条件の必須指定。オプションは[こちら](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)。
7. **Create Queue** ボタンをクリックします。


## Launch エージェントをセットアップする

以下では、エージェントをどこにデプロイするか、デプロイ先に応じてどう設定するかを説明します。

Amazon SageMaker 用の Launch エージェントのデプロイ方法には[いくつかの選択肢]({{< relref path="#decide-where-to-run-the-launch-agent" lang="ja" >}})があります。ローカルマシン、EC2 インスタンス、または EKS クラスターです。デプロイ先に応じて、[Launch エージェントを適切に設定]({{< relref path="#configure-a-launch-agent" lang="ja" >}}) してください。


### Launch エージェントの実行先を決める

プロダクションのワークロードや、すでに EKS クラスターをお持ちのお客様には、Helm チャートを使って EKS クラスターに Launch エージェントをデプロイすることを推奨します。

現時点で EKS クラスターがないプロダクションのワークロードには、EC2 インスタンスが良い選択です。Launch エージェントのインスタンスは常時稼働しますが、`t2.micro` 程度の手頃なインスタンスで十分です。

検証や個人のユースケースなら、ローカルマシンで Launch エージェントを実行するのが最速の始め方です。

ユースケースに応じて、以下のタブの手順に従って Launch エージェントを正しく設定してください。 
{{< tabpane text=true >}}
{{% tab "EKS" %}}
W&B は、EKS クラスターにエージェントをインストールする際、[W&B managed helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) の使用を強く推奨します。
{{% /tab %}}
{{% tab "EC2" %}}
Amazon EC2 ダッシュボードに移動し、以下を実施します。

1. **Launch instance** をクリックします。
2. **Name** フィールドに名前を入力します。必要に応じてタグを追加します。
2. **Instance type** から EC2 コンテナのインスタンスタイプを選択します。1vCPU と 1GiB メモリ（例: t2.micro）で十分です。 
3. **Key pair (login)** フィールドで組織用のキーペアを作成します。後でこのキーペアを使って SSH クライアントで [EC2 インスタンスに接続](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html) します。
2. **Network settings** で、組織に適したセキュリティグループを選択します。 
3. **Advanced details** を展開します。**IAM instance profile** で、先ほど作成した Launch エージェントの IAM ロールを選択します。
2. **Summary** を確認し、問題なければ **Launch instance** を選択します。 

AWS の EC2 ダッシュボード左側パネルの **Instances** に移動します。作成した EC2 インスタンスが実行中であること（**Instance state** 列）を確認します。EC2 インスタンスの起動を確認したら、ローカルマシンのターミナルに移動し、次を実施します。

1. **Connect** を選択します。 
2. **SSH client** タブを選び、表示された手順に従って EC2 インスタンスに接続します。
3. EC2 インスタンス内で、次のパッケージをインストールします:
```bash
sudo yum install python311 -y && python3 -m ensurepip --upgrade && pip3 install wandb && pip3 install wandb[launch]
```
4. 次に、EC2 インスタンス内で Docker をインストールして起動します:
```bash
sudo yum update -y && sudo yum install -y docker python3 && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -a -G docker ec2-user

newgrp docker
```

これで Launch エージェントの設定に進めます。

{{% /tab %}}
{{% tab "Local machine" %}}

ローカルマシンでポーリングするエージェントにロールを関連付けるには、`~/.aws/config` と `~/.aws/credentials` にある AWS の設定ファイルを使用します。前のステップで作成した Launch エージェント用の IAM ロール ARN を指定してください。
 
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

セッショントークンの有効期間は、紐づくプリンシパルに応じて [最大](https://docs.aws.amazon.com/cli/latest/reference/sts/get-session-token.html#description) 1 時間または 3 日です。
{{% /tab %}}
{{< /tabpane >}}


### Launch エージェントを設定する
`launch-config.yaml` という名前の YAML 設定ファイルで Launch エージェントを設定します。 

デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` に設定ファイルがあるか確認します。`-c` フラグでエージェントを起動するときに、別の ディレクトリー を指定することもできます。

以下の YAML は、エージェントのコア設定オプションの指定例です:

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


 ## （任意）Launch ジョブの Docker イメージを Amazon ECR にプッシュする

{{% alert %}}
このセクションは、トレーニングまたは推論ロジックを含む既存の Docker イメージを Launch エージェントが使用する場合にのみ該当します。[Launch エージェントの動作には 2 つの選択肢があります。]({{< relref path="#decide-if-you-want-the-launch-agent-to-build-a-docker-images" lang="ja" >}})  
{{% /alert %}}

Launch ジョブを含む Docker イメージを Amazon ECR のリポジトリにアップロードします。イメージベースのジョブを使用する場合は、新しい Launch ジョブを送信する前に、Docker イメージが ECR レジストリに存在している必要があります。