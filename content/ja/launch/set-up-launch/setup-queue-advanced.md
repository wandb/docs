---
title: Configure launch queue
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-queue-advanced
    parent: set-up-launch
url: guides/launch/setup-queue-advanced
---

ローンチキューオプションの設定方法について説明します。

## キュー設定テンプレートのセットアップ
Queue Config Templates を使用して、コンピュータの消費に対するガードレールを管理し、デフォルト、最小値、および最大値を設定します。メモリ使用量、GPU、実行時間などのフィールドに対するデフォルト、最小値、最大値を設定します。

設定テンプレートでキューを構成した後、チームのメンバーは設定された範囲内でのみ定義されたフィールドを変更できます。

### キューテンプレートの設定
既存のキュー上でキューテンプレートを構成するか、新しいキューを作成することができます。

1. [https://wandb.ai/launch](https://wandb.ai/launch) のLaunchアプリに移動します。
2. テンプレートを追加したいキューの名前の横にある **View queue** を選択します。
3. **Config** タブを選択します。ここにはキューの作成時、キュー設定、既存のローンチタイムオーバーライドなどの情報が表示されます。
4. **Queue config** セクションに移動します。
5. テンプレートを作成するための設定キーと値を特定します。
6. 設定内の値をテンプレートフィールドに置き換えます。テンプレートフィールドは `{{variable-name}}` の形式をとります。
7. **Parse configuration** ボタンをクリックします。設定を解析すると、W&B は設定したそれぞれのテンプレートのタイルを自動的に作成します。
8. 生成された各タイルに対して、まずキュー設定が許可するデータ型（文字列、整数、または浮動小数点数）を指定する必要があります。そのためには、**Type** ドロップダウンメニューからデータ型を選択します。
9. データ型に基づいて、各タイル内に表示されるフィールドに入力します。
10. **Save config** をクリックします。

例えば、AWSインスタンスをチームが使用できるものに制限するテンプレートを作成したいとします。テンプレートフィールドを追加する前のキュー設定は次のようになります：

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

`InstanceType` 用にテンプレートフィールドを追加すると、設定は次のようになります：

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

次に、**Parse configuration** をクリックします。新しい `aws-instance` とラベル付けされたタイルが **Queue config** の下に表示されます。

そこから、**Type** ドロップダウンからデータ型として String を選択します。これにより、ユーザーが選択できる値を指定するフィールドが入力されます。たとえば、次の画像のように、チームの管理者がユーザーが選択できる二つの異なるAWSインスタンスタイプを設定しました（`ml.m4.xlarge`と`ml.p3.xlarge`）:

{{< img src="/images/launch/aws_template_example.png" alt="" >}}



## ローンチジョブを動的に設定する
キュー設定は、エージェントがキューからジョブをデキューするときに評価されるマクロを使用して動的に設定できます。次のマクロを設定できます：

| Macro             | Description                                              |
|-------------------|----------------------------------------------------------|
| `${project_name}` | run がローンチされるプロジェクトの名前。                |
| `${entity_name}`  | run がローンチされるプロジェクトのオーナー。            |
| `${run_id}`       | ローンチされる run のID。                              |
| `${run_name}`     | ローンチされる run の名前。                             |
| `${image_uri}`    | この run のコンテナイメージのURI。                      |

{{% alert %}}
前述の表に記載されていないカスタムマクロ（例: `${MY_ENV_VAR}`）は、エージェントの環境からの環境変数で置き換えられます。
{{% /alert %}}

## ローンチエージェントを使用してアクセラレータ（GPU）で実行されるイメージを構築する
ローンチを使用してアクセラレータ環境で実行されるイメージを構築する場合、アクセラレータベースイメージを指定する必要があるかもしれません。

このアクセラレータベースイメージは、次の要件を満たす必要があります：

- Debian 互換性（ローンチの Dockerfile は apt-get を使用して python を取得します）
- CPU & GPU ハードウェア命令セットの互換性（使用する予定の GPU がサポートしている CUDA バージョンを確認してください）
- 提供するアクセラレータバージョンと ML アルゴリズムにインストールされるパッケージとの互換性
- ハードウェアと互換性を設定するために追加のステップを要求するパッケージ

### TensorFlow で GPU を使用する方法

TensorFlow が正しく GPU を利用するようにします。これを実行するには、キューリソース設定で `builder.accelerator.base_image` キーの Docker イメージとそのイメージタグを指定します。

例えば、`tensorflow/tensorflow:latest-gpu` ベースイメージは TensorFlow が GPU を正しく利用することを保証します。これは、キュー内のリソース設定を使用して設定できます。

以下の JSON スニペットは、キュー設定で TensorFlow のベースイメージを指定する方法を示しています:

```json title="Queue config"
{
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```