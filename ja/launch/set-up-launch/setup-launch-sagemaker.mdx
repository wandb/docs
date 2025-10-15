---
title: 'チュートリアル: SageMaker で W&B Launch を設定する'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-sagemaker
    parent: set-up-launch
url: /ja/guides/launch/setup-launch-sagemaker
---

W&B Launch を使用して、提供されたアルゴリズムやカスタムアルゴリズムを使用して SageMaker プラットフォーム上で機械学習モデルをトレーニングするための ラーンンチ ジョブを Amazon SageMaker に送信できます。SageMaker はコンピュート リソースの立ち上げとリリースを担当するため、EKS クラスターを持たないチームには良い選択肢となります。

Amazon SageMaker に接続された W&B Launch キューに送信された ラーンンチ ジョブは、[CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) を使用して SageMaker トレーニング ジョブとして実行されます。 CreateTrainingJob `API` に送信される引数を制御するには、 ラーンンチ キュー設定 を使用します。

Amazon SageMaker は [トレーニング ジョブを実行するために Docker イメージを使用しています](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html)。SageMaker によってプルされるイメージは、Amazon Elastic Container Registry (ECR) に保存する必要があります。 つまり、トレーニングに使用するイメージは ECR に保存する必要があります。

{{% alert %}}
このガイドでは、SageMaker トレーニング ジョブを実行する方法を示しています。Amazon SageMaker での推論用にモデルを展開する方法については、[この例の Launch ジョブ](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints) を参照してください。
{{% /alert %}}

## 前提条件 

始める前に、以下の前提条件を確認してください:

* [Docker イメージを作成するかどうかを決定します]({{< relref path="#decide-if-you-want-the-launch-agent-to-build-a-docker-images" lang="ja" >}})。
* [AWS リソースを設定し、S3、ECR、および Sagemaker IAM ロールに関する情報を収集します]({{< relref path="#set-up-aws-resources" lang="ja" >}})。
* [Launch エージェントのための IAM ロールを作成します]({{< relref path="#create-an-iam-role-for-launch-agent" lang="ja" >}})。

### Docker イメージを作成するかどうかを決定する

W&B Launch エージェントに Docker イメージを作成させるかどうかを決定します。選択肢は 2 つあります。

* ローンンチ エージェントに Docker イメージの構築を許可し、Amazon ECR にイメージをプッシュし、[SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) ジョブの送信を許可します。このオプションは、トレーニング コードを迅速に反復する ML エンジニアにいくらかの簡素化を提供できます。
* ローンンチ エージェントが、トレーニングまたは推論スクリプトを含む既存の Docker イメージを使用します。このオプションは既存の CI システムに適しています。このオプションを選択する場合は、Amazon ECR のコンテナ レジストリに Docker イメージを手動でアップロードする必要があります。

### AWS リソースを設定する

お好みの AWS リージョンで次の AWS リソースが設定されていることを確認してください :

1. コンテナ イメージを保存するための [ECR リポジトリ](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html)。
2. SageMaker トレーニング ジョブの入力と出力を保存するための 1 つまたは複数の [S3 バケット](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)。
3. Amazon SageMaker がトレーニング ジョブを実行し、Amazon ECR と Amazon S3 と対話することを許可する IAM ロール。

これらのリソースの ARN をメモしておいてください。SageMaker 用に [Launch キュー設定]({{< relref path="#configure-launch-queue-for-sagemaker" lang="ja" >}}) を定義するときに ARN が必要になります。

### Launch エージェント用の IAM ポリシーを作成する

1. AWS の IAM 画面から、新しいポリシーを作成します。
2. JSON ポリシーエディターに切り替え、以下のポリシーをケースに基づいて貼り付けます。`<>` で囲まれた値を実際の値に置き換えてください:

{{< tabpane text=true >}}
{{% tab "エージェントが事前構築された Docker イメージを送信" %}}
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
{{% tab "エージェントが Docker イメージを構築して送信" %}}
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

3. **次へ** をクリックします。
4. ポリシーに名前と説明を付けます。
5. **ポリシー作成** をクリックします。

### Launch エージェント用の IAM ロールを作成する

Launch エージェントには、Amazon SageMaker トレーニング ジョブを作成する権限が必要です。以下の手順に従って IAM ロールを作成します:

1. AWS の IAM 画面から、新しいロールを作成します。
2. **信頼されたエンティティ** として **AWS アカウント** (または組織のポリシーに適したオプション) を選択します。
3. 権限画面をスクロールし、上で作成したポリシー名を選択します。
4. ロールに名前と説明を付けます。
5. **ロールの作成** を選択します。
6. ロールの ARN を記録します。これを設定するときに Launch エージェント用に ARN を指定します。

IAM ロールの作成方法について詳しくは、[AWS Identity and Access Management ドキュメント](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html) を参照してください。

{{% alert %}}
* エージェントがイメージを構築できるようにするには、[高度なエージェントの設定]({{< relref path="./setup-agent-advanced.md" lang="ja" >}})で追加の権限が必要です。
* SageMaker キューの `kms:CreateGrant` 権限は、関連する ResourceConfig に指定された VolumeKmsKeyId がある場合にのみ必要であり、関連するロールにこの操作を許可するポリシーがない場合に限ります。
{{% /alert %}}

## SageMaker 用に Launch キューを設定する

次に、W&B アプリで SageMaker をコンピュート リソースとして使用するキューを作成します:

1. [Launch アプリ](https://wandb.ai/launch) に移動します。
2. **キューを作成** ボタンをクリックします。
4. キューを作成する **エンティティ** を選択します。
5. **名前** フィールドにキューの名前を入力します。
6. **リソース** として **SageMaker** を選択します。
7. **設定** フィールド内で、SageMaker ジョブに関する情報を提供します。デフォルトでは、W&B は YAML および JSON の `CreateTrainingJob` リクエストボディを自動生成します:
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
少なくとも以下を指定する必要があります :

- `RoleArn` : SageMaker 実行 IAM ロールの ARN ([前提条件]({{< relref path="#prerequisites" lang="ja" >}}) を参照してください)。Launch **agent** IAM ロールとは混同しないでください。
- `OutputDataConfig.S3OutputPath` : SageMaker の出力が保存される Amazon S3 URI を指定します。
- `ResourceConfig`: リソース設定の必須仕様です。リソース設定のオプションは[こちら](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceConfig.html)に記載されています。
- `StoppingCondition`: トレーニング ジョブの停止条件の必須仕様です。オプションは[こちら](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)に記載されています。
8. **キューを作成** ボタンをクリックします。

## Launch エージェントをセットアップする

次のセクションでは、エージェントをデプロイする場所と、デプロイ場所に基づいてエージェントをどのように設定するかを説明します。

Amazon SageMaker キューに Launch エージェントをデプロイする方法には[いくつかのオプションがあります]({{< relref path="#decide-where-to-run-the-launch-agent" lang="ja" >}}): ローカルマシン、EC2 インスタンス、または EKSクラスターで。エージェントをデプロイする場所に基づいて[アプリケーション エージェントを適切に構成します]({{< relref path="#configure-a-launch-agent" lang="ja" >}})。

### ローンンチ エージェントを実行する場所を決定する

プロダクション ワークロードおよび既に EKS クラスターを持つ顧客には、この Helm チャートを使用して EKS クラスターに ラーンンチ エージェント をデプロイすることをお勧めします。

現在の EKS クラスターがないプロダクション ワークロードには、EC2 インスタンスが適したオプションです。Launch エージェント インスタンスは常に稼働していますが、`t2.micro` サイズの EC2 インスタンスという比較的手頃なインスタンスで十分です。

実験的または個人のユースケースには、ローカルマシンに Launch エージェントを実行するのがすばやく始める方法です。

選択したユースケースに基づいて、以下のタブに記載されている指示に従って Launch エージェントを適切に設定してください: 
{{< tabpane text=true >}}
{{% tab "EKS" %}}
W&B は、エージェントを EKS クラスターでインストールするために、[W&B 管理 helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) の使用を強く推奨しています。
{{% /tab %}}
{{% tab "EC2" %}}
Amazon EC2 ダッシュボードに移動し、次のステップを完了します:

1. **インスタンスを起動** をクリックします。
2. **名前** フィールドに名前を入力します。タグをオプションで追加します。
3. **インスタンスタイプ** から、あなたの EC2 コンテナ用のインスタンスタイプを選択します。1vCPU と 1GiB のメモリ以上は必要ありません (例えば t2.micro)。
4. **キーペア（ログイン）** フィールドで、組織内の新しいキーペアを作成します。後のステップで選択した SSH クライアントで EC2 インスタンスに [接続する](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html) ために、このキーペアを使用します。
5. **ネットワーク設定** で、組織に適したセキュリティグループを選択します。
6. **詳細設定** を展開します。**IAM インスタンスプロファイル** として、上記で作成した ローンンチ エージェント IAM ロールを選択します。
7. **サマリー** フィールドを確認します。正しければ、**インスタンスを起動** を選択します。

AWS 上の EC2 ダッシュボードの左側パネル内の **インスタンス** に移動します。作成した EC2 インスタンスが稼働している ( **インスタンス状態** 列を参照) ことを確認します。EC2 インスタンスが稼働していることを確認したら、ローカルマシンのターミナルに移動し、次の手順を完了します:

1. **接続** を選択します。
2. **SSH クライアント** タブを選択し、EC2 インスタンスに接続するための指示に従います。
3. EC2インスタンス内で、次のパッケージをインストールします:
```bash
sudo yum install python311 -y && python3 -m ensurepip --upgrade && pip3 install wandb && pip3 install wandb[launch]
```
4. 次に、EC2 インスタンス内に Docker をインストールして起動します:
```bash
sudo yum update -y && sudo yum install -y docker python3 && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -a -G docker ec2-user

newgrp docker
```

これで、Launchエージェントの構成を設定する準備が整いました。

{{% /tab %}}
{{% tab "ローカルマシン" %}}

ローカルマシンでポーリングを実行するエージェントとロールを関連付けるには、`~/.aws/config` と `~/.aws/credentials` にある AWS 設定ファイルを使用します。前のステップで作成した Launch エージェントの IAM ロール ARN を指定します。

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

セッショントークンは、その主データと関連付けられた AWS リソースによって[最大長](https://docs.aws.amazon.com/cli/latest/reference/sts/get-session-token.html#description)が 1 時間または 3 日であることに注意してください。
{{% /tab %}}
{{< /tabpane >}}

### Launch エージェントを設定する 

`launch-config.yaml` という名前の YAML 設定ファイルで Launch エージェントを設定します。

デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` にある設定ファイルを確認します。エージェントをアクティブにする際に `-c` フラグで別のディレクトリを指定することも可能です。

以下の YAML スニペットは、コア設定エージェントオプションを指定する方法を示しています:

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

エージェントは `wandb launch-agent` で開始します。

## (オプション) Docker イメージを Amazon ECR にプッシュする

{{% alert %}}
このセクションは、トレーニングまたは推論ロジックを含む既存の Docker イメージをエージェントが使用する場合にのみ適用されます。[Launch エージェントの動作には 2 つのオプションがあります。]({{< relref path="#decide-if-you-want-the-launch-agent-to-build-a-docker-images" lang="ja" >}})  
{{% /alert %}}

Launch ジョブを含む Docker イメージを Amazon ECR レポジトリにアップロードします。画像ベースのジョブを使用している場合、Docker イメージは新しい Launch ジョブを送信する前に ECR レジストリに存在している必要があります。