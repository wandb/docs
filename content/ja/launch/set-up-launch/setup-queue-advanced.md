---
title: ローンチキューを設定
menu:
  launch:
    identifier: setup-queue-advanced
    parent: set-up-launch
url: guides/launch/setup-queue-advanced
---

以下のページでは、ローンチキューのオプション設定について説明します。

## キュー設定テンプレートのセットアップ
Queue Config Templates を使って、計算リソース消費のガードレールの管理・運用を行えます。メモリ消費量、GPU、実行時間などのフィールドに対して、デフォルト値、最小値、最大値を設定できます。

設定テンプレートでキューを設定した後は、チームメンバーは指定した範囲内でのみ、定義したフィールドを変更できます。

### キューテンプレートの設定
既存のキュー上でキューテンプレートを設定するか、新たなキューを作成して設定できます。

1. [W&B Launch App](https://wandb.ai/launch) にアクセスします。
2. テンプレートを追加したいキュー名の横にある **View queue** を選択します。
3. **Config** タブを選択します。ここでキュー作成日時や現在のキュー設定、既存のローンチ時の上書き設定など情報が表示されます。
4. **Queue config** セクションに移動します。
5. テンプレート化したい設定のキーと値を特定します。
6. 設定内の値をテンプレートフィールド（例：`{{variable-name}}`）に置き換えます。
7. **Parse configuration** ボタンをクリックします。設定をパースすることで、作成した各テンプレートに対してキュー設定の下にタイルが自動生成されます。
8. 生成された各タイルごとに、キューが許容するデータ型（string、integer、float）をまず指定します。**Type** ドロップダウンからデータ型を選択してください。
9. 選択したデータ型に基づいて、各タイル内で必要なフィールドを入力します。
10. **Save config** をクリックします。

例として、チームが使用できる AWS インスタンスを制限するテンプレートを作成するとします。テンプレート用フィールドを追加する前のキュー設定は、次のようになっています。

```yaml title="launch config"
RoleArn: arn:aws:iam:region:account-id:resource-type/resource-id
ResourceConfig:
  InstanceType: ml.m4.xlarge
  InstanceCount: 1
  VolumeSizeInGB: 2
OutputDataConfig:
  S3OutputPath: s3://bucketname
StoppingCondition:
  MaxRuntimeInSeconds: 3600
```

`InstanceType` にテンプレートフィールドを追加すると、設定は次のようになります。

```yaml title="launch config"
RoleArn: arn:aws:iam:region:account-id:resource-type/resource-id
ResourceConfig:
  InstanceType: "{{aws_instance}}"
  InstanceCount: 1
  VolumeSizeInGB: 2
OutputDataConfig:
  S3OutputPath: s3://bucketname
StoppingCondition:
  MaxRuntimeInSeconds: 3600
```

その後、**Parse configuration** をクリックします。**Queue config** の下に `aws-instance` というラベルの新しいタイルが表示されます。

ここで **Type** ドロップダウンから String を選択します。ユーザーが指定できる値を設定できるフィールドが入力されます。下記の画像では、チーム管理者が利用可能な2種類の AWS インスタンスタイプ（`ml.m4.xlarge` と `ml.p3.xlarge`）を設定しています。

{{< img src="/images/launch/aws_template_example.png" alt="AWS CloudFormation template" >}}



## ローンチジョブを動的に設定する
Queue config には、エージェントがキューからジョブを取得する際に評価されるマクロを利用して動的に設定できます。以下のマクロが使用可能です。

| マクロ              | 説明                                               |
|-------------------|---------------------------------------------------|
| `${project_name}` | 実行時の Project 名です。                         |
| `${entity_name}`  | 実行先 Project のオーナー名です。                 |
| `${run_id}`       | Launch される Run の ID です。                    |
| `${run_name}`     | Launch される Run の名前です。                    |
| `${image_uri}`    | この Run 用のコンテナイメージの URI です。        |

{{% alert %}}
上記リストにないカスタムマクロ（例：`${MY_ENV_VAR}`）は、エージェントの環境から対応する環境変数値に置き換えられます。
{{% /alert %}}

## ローンチエージェントで GPU 対応のイメージをビルドする
ローンチでアクセラレータ（GPU）環境向けイメージをビルドする場合は、アクセラレータベースイメージの指定が必要になることがあります。

このアクセラレータベースイメージは以下の条件を満たしている必要があります。

- Debian 互換性（Launch の Dockerfile では apt-get で python を取得します）
- CPU・GPU ハードウェア命令セットの互換性（利用予定 GPU がサポートする CUDA バージョンであること）
- 指定するアクセラレータバージョンと ML アルゴリズムでインストールされるパッケージとが互換であること
- ハードウェア用の追加セットアップが必要なパッケージは事前にインストールされていること

### TensorFlow で GPU を使用する方法

TensorFlow が GPU を正しく利用できるようにするには、キューリソース設定の `builder.accelerator.base_image` キーで使用する Docker イメージとそのタグを指定してください。

例えば、`tensorflow/tensorflow:latest-gpu` ベースイメージを使うと TensorFlow が GPU 利用に対応できます。これはキューのリソース設定で指定できます。

以下の JSON スニペットは、キュー設定で TensorFlow ベースイメージを指定する方法を示しています。

```json title="Queue config"
{
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```