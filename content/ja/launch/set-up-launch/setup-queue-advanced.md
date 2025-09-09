---
title: Launch キューの設定
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-queue-advanced
    parent: set-up-launch
url: guides/launch/setup-queue-advanced
---

このページでは、Launch キューのオプションを設定する方法を説明します。

## キュー設定テンプレートをセットアップする
Queue Config Templates を使って、計算リソース消費に対するガードレールを管理・運用できます。メモリ使用量、GPU、実行時間などのフィールドに対して、デフォルト値、最小値、最大値を設定できます。

キューに設定テンプレートを設定すると、チームメンバーは指定した範囲内でのみ、あなたが定義したフィールドを変更できます。

### キューテンプレートを設定する
既存のキューにテンプレートを設定することも、新しいキューを作成して設定することもできます。  

1. [W&B Launch App](https://wandb.ai/launch) に移動します。
2. テンプレートを追加したいキュー名の横にある **View queue** を選択します。
3. **Config** タブを選択します。ここにはキューの作成日時、キューの設定、既存の起動時オーバーライドなど、キューに関する情報が表示されます。
4. **Queue config** セクションに移動します。
5. テンプレートを作成したい設定のキーと値を特定します。
6. 設定内の値をテンプレートフィールドに置き換えます。テンプレートフィールドは `{{variable-name}}` の形式を取ります。
7. **Parse configuration** ボタンをクリックします。設定をパースすると、作成した各テンプレートに対応するタイルがキュー設定の下に自動的に作成されます。
8. 生成された各タイルについて、まずキュー設定で許可するデータ型 (string、integer、float) を指定する必要があります。**Type** ドロップダウンメニューからデータ型を選択してください。
9. 選択したデータ型に基づいて、各タイル内に表示されるフィールドを入力します。
10. **Save config** をクリックします。

たとえば、チームが使用できる AWS インスタンスを制限するテンプレートを作成したいとします。テンプレートフィールドを追加する前のキュー設定は次のようになります:

```yaml title="Launch 設定"
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

`InstanceType` にテンプレートフィールドを追加すると、設定は次のようになります:

```yaml title="Launch 設定"
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

次に **Parse configuration** をクリックします。**Queue config** の下に `aws-instance` とラベル付けされた新しいタイルが表示されます。

そこから **Type** ドロップダウンでデータ型として String を選択します。すると、ユーザーが選択できる値を指定するためのフィールドが表示されます。たとえば、次の画像ではチームの管理者が、ユーザーが選択できる 2 種類の AWS インスタンスタイプ (`ml.m4.xlarge` と `ml.p3.xlarge`) を設定しています:

{{< img src="/images/launch/aws_template_example.png" alt="AWS CloudFormation テンプレート" >}}

## Launch ジョブを動的に設定する
キュー設定は、エージェントがキューからジョブをデキューする際に評価されるマクロを使って動的に設定できます。使用できるマクロは次のとおりです:

| マクロ              | 説明                                            |
|-------------------|-------------------------------------------------|
| `${project_name}` | run が起動される project の名前。               |
| `${entity_name}`  | run が起動される project の所有者。             |
| `${run_id}`       | 起動される run の ID。                          |
| `${run_name}`     | 起動中の run の名前。                            |
| `${image_uri}`    | この run 用コンテナイメージの URI。              |

{{% alert %}}
上の表にないカスタムマクロ (たとえば `${MY_ENV_VAR}`) は、エージェントの環境から同名の環境変数で置き換えられます。
{{% /alert %}}

## アクセラレータ (GPU) 上で実行するイメージをビルドするために Launch エージェントを使う
アクセラレータ環境で実行されるイメージを Launch でビルドする場合、アクセラレータ用のベースイメージを指定する必要があることがあります。

このアクセラレータ用ベースイメージは、次の要件を満たす必要があります:

- Debian 互換性 (Launch Dockerfile は apt-get を使って Python を取得します)
- CPU と GPU のハードウェア命令セットとの互換性 (使用する GPU が該当の CUDA バージョンをサポートしていることを確認してください)
- 提供するアクセラレータのバージョンと、ML アルゴリズムで使用するパッケージとの互換性
- ハードウェアとの互換性を確保するための追加設定が必要なパッケージがインストールされていること

### TensorFlow で GPU を使う方法

TensorFlow が GPU を適切に活用するようにします。そのために、キューのリソース設定で `builder.accelerator.base_image` キーに対して Docker イメージとそのイメージタグを指定します。

たとえば、`tensorflow/tensorflow:latest-gpu` ベースイメージを指定すると、TensorFlow が GPU を適切に使用します。これはキューのリソース設定で構成できます。

次の JSON スニペットは、キュー設定で TensorFlow のベースイメージを指定する方法を示しています:

```json title="Queue 設定"
{
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```