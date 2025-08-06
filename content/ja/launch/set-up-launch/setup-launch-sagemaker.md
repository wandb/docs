---
title: 'チュートリアル: SageMaker で W&B Launch をセットアップする'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-sagemaker
    parent: set-up-launch
url: guides/launch/setup-launch-sagemaker
---

W&B Launch を使うと、Amazon SageMaker 上で提供アルゴリズムやカスタムアルゴリズムを利用して機械学習モデルのトレーニング用 launch ジョブを送信できます。SageMaker はコンピュートリソースの起動と解放を自動で管理するため、EKS クラスターがなくてもチームで簡単に使えます。

Amazon SageMaker に接続された W&B Launch キューに送られた launch ジョブは、[CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) を使って SageMaker Training Job として実行されます。launch キューの設定で、`CreateTrainingJob` API に渡す引数をコントロールします。

Amazon SageMaker は[Docker イメージを利用してトレーニングジョブを実行します](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html)。SageMaker が取得するイメージは Amazon Elastic Container Registry (ECR) 上に格納する必要があります。つまり、トレーニングで使用するイメージは ECR に保存してください。

{{% alert %}}
このガイドは SageMaker Training Job の実行方法を解説しています。Amazon SageMaker にモデルをデプロイして推論を行う方法については、[こちらの Launch ジョブ例](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints)をご確認ください。
{{% /alert %}}

## 前提条件

始める前に、以下の前提条件を満たしているかご確認ください：

* [Launch agent に Docker イメージのビルドを任せたいか選択します。]({{< relref path="#decide-if-you-want-the-launch-agent-to-build-a-docker-images" lang="ja" >}})
* [AWS リソースのセットアップ、S3・ECR・Sagemaker IAM ロールなどの情報取得。]({{< relref path="#set-up-aws-resources" lang="ja" >}})
* [Launch agent のための IAM ロールを作成]({{< relref path="#create-an-iam-role-for-launch-agent" lang="ja" >}})。

### Launch agent に Docker イメージをビルドさせるか決める

W&B Launch agent に Docker イメージのビルドを任せるか選択してください。以下の2つのオプションがあります：

* launch agent が Docker イメージをビルドし ECR へ push、そのイメージで [SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html) ジョブを送信します。この方法はトレーニングコードを素早く反復したい ML エンジニアにとってシンプルな選択肢です。
* すでに構築済みの Docker イメージ（トレーニング・推論スクリプトを含む）を launch agent が使う。このオプションは既存の CI システムとの連携にも適しています。選択した場合は、ご自身で Docker イメージを ECR レジストリにアップロードしてください。

### AWS リソースのセットアップ

任意の AWS リージョンで下記リソースを用意してください：

1. コンテナイメージを保存するための [ECR リポジトリー](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html)
2. SageMaker Training ジョブで使う入力・出力データを保存するための [S3 バケット](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)（1つ以上）
3. SageMaker がトレーニングジョブを実行し、ECR ・ S3 とやりとりできるようにするための IAM ロール

これらリソースの ARN（Amazon Resource Name）を控えておいてください。[Launch queue 設定]({{< relref path="#configure-launch-queue-for-sagemaker" lang="ja" >}}) で ARN が必要になります。

### Launch agent 用 IAM ポリシーを作成する

1. AWS の IAM 画面から新しいポリシーを作成します。
2. JSON ポリシーエディタに切り替え、用途に応じて下記ポリシーを貼り付けてください。`<>` で囲まれている値はご自身の値に置き換えます：

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

### Launch agent 用 IAM ロールを作成

Launch agent には、Amazon SageMaker トレーニングジョブの作成権限が必要です。下記手順で IAM ロールを作成します：

1. AWS の IAM 画面から新しいロールを作成します。
2. **Trusted Entity** には **AWS Account**（組織のポリシーによって別の選択肢も可）を選択してください。
3. 権限画面で、先ほど作成したポリシー名を選択します。
4. ロールに名前・説明を付けます。
5. **Create role** を選択します。
6. 作成したロールの ARN を控えておきます。launch agent 設定時にこの ARN を使用します。

IAM ロールの作成については [AWS Identity and Access Management Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html) をご参照ください。

{{% alert %}}
* launch agent にイメージのビルドも任せたい場合は、追加の権限を含めた [高度な agent セットアップ]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) も参照してください。
* SageMaker キュー用 `kms:CreateGrant` パーミッションは、ResourceConfig で VolumeKmsKeyId が指定された場合かつロールのポリシーで明示的に許可していないときのみ必要です。
{{% /alert %}}


## SageMaker 用 launch queue を設定

次に、SageMaker を計算リソースに設定した Queue を W&B App 上で作成します：

1. [Launch App](https://wandb.ai/launch) にアクセスします。
3. **Create Queue** ボタンをクリックします。
4. キューを作成したい **Entity** を選択します。
5. **Name** 欄にキューの名前を入力します。
6. **Resource** として **SageMaker** を選択します。
7. **Configuration** 欄には SageMaker ジョブの情報を入力します。デフォルトで、W&B が `CreateTrainingJob` の YAML および JSON リクエストボディを自動入力します:
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
最低限、以下を指定してください：

- `RoleArn` : SageMaker 実行用 IAM ロールの ARN（[前提条件]({{< relref path="#prerequisites" lang="ja" >}})参照）。launch **agent** IAM ロールとは異なる点に注意してください。
- `OutputDataConfig.S3OutputPath` : SageMaker の出力が保存される Amazon S3 URI。
- `ResourceConfig`: リソース構成の必須指定。[こちら](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceConfig.html) で詳細確認できます。
- `StoppingCondition`: トレーニングジョブの停止条件の必須指定。[こちら](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html) をご参照ください。
7. **Create Queue** ボタンを押してください。

## launch agent をセットアップ

このセクションでは、agent のデプロイ先と、それに応じた agent の設定方法を解説します。

Amazon SageMaker 用 Launch agent の[デプロイ場所は複数選択肢があります]({{< relref path="#decide-where-to-run-the-launch-agent" lang="ja" >}})。
ローカルマシン、EC2 インスタンス、EKS クラスターのいずれかで agent を動かします。デプロイ先に合わせて[launch agent の設定]({{< relref path="#configure-a-launch-agent" lang="ja" >}})をしてください。

### Launch agent のデプロイ先を決める

プロダクション用途や既に EKS クラスターをお持ちのお客様には、この Helm chart を使った Launch agent の EKS クラスターへのデプロイを推奨しています。

EKS クラスターがまだない場合は、EC2 インスタンスを使った運用がおすすめです。launch agent インスタンスは常時稼働しますが、`t2.micro` 程度の安価なインスタンス（1vCPU/1GiBメモリ）で問題ありません。

検証や個人での利用の場合、ローカルマシン上で Launch agent を実行すると手軽に始められます。

用途に合わせて、以下タブの手順にしたがって launch agent を正しくセットアップしてください。 
{{< tabpane text=true >}}
{{% tab "EKS" %}}
EKS クラスターにエージェントをインストールするには [W&B 管理の helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) のご利用を強く推奨します。
{{% /tab %}}
{{% tab "EC2" %}}
Amazon EC2 ダッシュボードで下記手順を実施してください：

1. **Launch instance** をクリック
2. **Name** 欄に任意の名前を設定（必要に応じてタグ付け）
2. **Instance type** からインスタンスタイプを選択します。EC2 コンテナに必要なメモリや CPU は 1vCPU / 1GiB 以上あれば OK です（例: t2.micro）。
3. **Key pair (login)** フィールドで組織用のキーペアを作成します。後で [EC2 インスタンスに SSH 接続](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect.html) する際にこのキーペアを使用します。
2. **Network settings** で組織に適したセキュリティグループを選択します
3. **Advanced details** を展開し、**IAM instance profile** には上述した Launch agent IAM ロールを選択
2. **Summary** を確認し、問題なければ **Launch instance** としてください

AWS EC2 ダッシュボード左パネルの **Instances** を開き、作成した EC2 インスタンスが稼働状態 (**Instance state** 列) であることを確認してください。稼働状態になったら、ローカルマシンのターミナルから下記の手順を実施します。

1. **Connect** を選択
2. **SSH client** タブで接続手順を表示、案内に従い EC2 インスタンスに SSH 接続
3. EC2 インスタンス内で以下のパッケージをインストール：
```bash
sudo yum install python311 -y && python3 -m ensurepip --upgrade && pip3 install wandb && pip3 install wandb[launch]
```
4. 続いて Docker のインストール・起動を行います：
```bash
sudo yum update -y && sudo yum install -y docker python3 && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -a -G docker ec2-user

newgrp docker
```

あとは Launch agent 設定を進めてください。

{{% /tab %}}
{{% tab "Local machine" %}}

ローカルマシンで agent を動かしたい場合は、`~/.aws/config` および `~/.aws/credentials` の AWS 設定ファイルを利用し agent にロールを関連付けます。前のステップで作成した launch agent 用 IAM ロールの ARN を指定してください。

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

セッショントークンの [最大有効期間](https://docs.aws.amazon.com/cli/latest/reference/sts/get-session-token.html#description) は関連するプリンシパルによって 1 時間または 3 日間です。
{{% /tab %}}
{{< /tabpane >}}

### launch agent の設定

launch agent は `launch-config.yaml` という YAML 設定ファイルで設定します。

デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` で設定ファイルを探します。`-c` フラグで異なるディレクトリーを指定することも可能です。

下記はエージェントの主要オプションの指定例です：

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

`wandb launch-agent` コマンドで agent を起動してください。


 ## （オプション）Docker イメージを Amazon ECR へアップロード

{{% alert %}}
このセクションは、launch agent が学習や推論のロジックを内包した既存の Docker イメージを使用する場合のみ該当します。[launch agent の動作には 2 パターンあります。]({{< relref path="#decide-if-you-want-the-launch-agent-to-build-a-docker-images" lang="ja" >}})  
{{% /alert %}}

launch ジョブ用の Docker イメージを Amazon ECR リポジトリにアップロードしてください。イメージベースのジョブを使う場合は、新しい launch ジョブを送信する前にイメージが ECR レジストリに存在している必要があります。