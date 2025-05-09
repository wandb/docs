---
title: ローンンチキューを設定する
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-queue-advanced
    parent: set-up-launch
url: /ja/guides/launch/setup-queue-advanced
---

以下のページでは、ローンチキューオプションの設定方法について説明します。

## キュー設定テンプレートのセットアップ
Queue Config Templates を使用して、計算リソースの消費に関するガードレールを管理します。メモリ消費量、GPU、実行時間などのフィールドに対して、デフォルト、最小値、および最大値を設定します。

config templates を使用してキューを設定した後、チームのメンバーは、あなたが定義した範囲内のフィールドのみを変更することができます。

### キューテンプレートの設定
既存のキューでキューテンプレートを設定するか、新しいキューを作成することができます。

1. [https://wandb.ai/launch](https://wandb.ai/launch) のローンチアプリに移動します。
2. テンプレートを追加したいキューの名前の横にある **View queue** を選択します。
3. **Config** タブを選択します。これにより、キューの作成日時、キュー設定、および既存のローンチタイムオーバーライドに関する情報が表示されます。
4. **Queue config** セクションに移動します。
5. テンプレートを作成したい設定キー-値を特定します。
6. 設定内の値をテンプレートフィールドに置き換えます。テンプレートフィールドは `{{variable-name}}` の形式をとります。
7. **Parse configuration** ボタンをクリックします。設定を解析すると、W&B は作成した各テンプレートの下にキュー設定タイルを自動的に作成します。
8. 生成された各タイルに対して、キュー設定が許可できるデータ型 (文字列、整数、浮動小数点数) を最初に指定する必要があります。これを行うために、**Type** ドロップダウンメニューからデータ型を選択します。
9. データ型に基づいて、各タイル内に表示されるフィールドを完成させます。
10. **Save config** をクリックします。

例えば、チームが使用できる AWS インスタンスを制限するテンプレートを作成したい場合、テンプレートフィールドを追加する前のキュー設定は次のようになります：

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

`InstanceType` にテンプレートフィールドを追加すると、設定は次のようになります：

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

次に、**Parse configuration** をクリックします。新しいタイル `aws-instance` が **Queue config** の下に表示されます。

そこで、**Type** ドロップダウンからデータ型として String を選択します。これにより、ユーザーが選択できる値を指定できるフィールドが表示されます。例えば、次の画像では、チームの管理者がユーザーが選べる 2 つの異なる AWS インスタンスタイプ (`ml.m4.xlarge` と `ml.p3.xlarge`) を設定しています：

{{< img src="/images/launch/aws_template_example.png" alt="" >}}

## ローンチジョブを動的に設定する
キュー設定は、エージェントがキューからジョブをデキューするときに評価されるマクロを使用して動的に設定できます。以下のマクロを設定できます：

| マクロ             | 説明                                           |
|-------------------|-------------------------------------------------------|
| `${project_name}` | run がローンチされるプロジェクトの名前。 |
| `${entity_name}`  | run がローンチされるプロジェクトの所有者。   |
| `${run_id}`       | ローンチされる run の ID。                     |
| `${run_name}`     | ローンチされる run の名前。                |
| `${image_uri}`    | この run のコンテナイメージの URI。          |

{{% alert %}}
前の表に記載されていないカスタムマクロ (例えば `${MY_ENV_VAR}`) は、エージェントの環境から環境変数で置き換えられます。
{{% /alert %}}

## アクセラレータ (GPU) で実行されるイメージをビルドするためのローンチエージェントの使用
アクセラレータ環境で実行されるイメージをビルドするためにローンチを使用する場合、アクセラレータベースイメージを指定する必要があります。

このアクセラレータベースイメージは次の要件を満たしている必要があります：

- Debian 互換 (Launch Dockerfile は python を取得するために apt-get を使用します)
- CPU & GPU ハードウェアインストラクションセットとの互換性 (使用する GPU がサポートする CUDA バージョンであることを確認してください)
- あなたが提供するアクセラレータバージョンと ML アルゴリズムにインストールされたパッケージ間の互換性
- ハードウェアとの互換性を確立するために必要な追加ステップを要求するパッケージのインストール

### TensorFlow で GPU を使用する方法

TensorFlow が GPU を適切に利用することを確認してください。これを達成するために、キューリソース設定の `builder.accelerator.base_image` キーで Docker イメージとそのイメージタグを指定します。

例えば、`tensorflow/tensorflow:latest-gpu` ベースイメージは、TensorFlow が GPU を適切に使用することを保証します。これはキュー設定でリソース設定を使用して設定できます。

以下の JSON スニペットは、キュー設定で TensorFlow ベースイメージを指定する方法を示しています：

```json title="Queue config"
{
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```