---
title: Configure launch queue
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-queue-advanced
    parent: set-up-launch
url: guides/launch/setup-queue-advanced
---

以下のページでは、 ローンチ キューのオプションを設定する方法について説明します。

## キュー設定テンプレートの設定

キュー設定テンプレートを使用して、コンピュート消費に関するガードレールを管理します。メモリ消費量、 GPU 、ランタイム時間などのフィールドのデフォルト値、最小値、および最大値を設定します。

設定テンプレートでキューを設定すると、チームのメンバーは、定義した範囲内でのみ、定義したフィールドを変更できます。

### キューテンプレートの設定

既存のキューでキューテンプレートを設定するか、新しいキューを作成できます。

1. [https://wandb.ai/launch](https://wandb.ai/launch) の ローンチ アプリに移動します。
2. テンプレートを追加するキューの名前の横にある **View queue** を選択します。
3. **Config** タブを選択します。これにより、キューが作成された時期、キューの設定、既存の ローンチ 時のオーバーライドなど、キューに関する情報が表示されます。
4. **Queue config** セクションに移動します。
5. テンプレートを作成する設定の キー の 値 を特定します。
6. 設定内の 値 をテンプレートフィールドに置き換えます。テンプレートフィールドは `{{variable-name}}` の形式を取ります。
7. **Parse configuration** ボタンをクリックします。設定を解析すると、作成した各テンプレートのタイルが自動的にキュー設定の下に作成されます。
8. 生成された各タイルについて、最初にキュー設定で許可する データ 型（文字列、整数、または浮動小数点）を指定する必要があります。これを行うには、**Type** ドロップダウンメニューから データ 型を選択します。
9. データ 型に基づいて、各タイル内に表示されるフィールドに入力します。
10. **Save config** をクリックします。

たとえば、チームが使用できる AWS インスタンスを制限するテンプレートを作成するとします。テンプレートフィールドを追加する前は、キュー設定は次のようになります。

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

`InstanceType` のテンプレートフィールドを追加すると、設定は次のようになります。

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

次に、**Parse configuration** をクリックします。`aws-instance` というラベルの新しいタイルが **Queue config** の下に表示されます。

そこから、**Type** ドロップダウンから String を データ 型として選択します。これにより、 ユーザー が選択できる 値 を指定できるフィールドが入力されます。たとえば、次の図では、チームの管理者が ユーザー が選択できる2つの異なる AWS インスタンスタイプ（`ml.m4.xlarge` と `ml.p3.xlarge`）を設定しています。

{{< img src="/images/launch/aws_template_example.png" alt="" >}}

## ローンチ ジョブの動的な設定

キュー設定は、 エージェント がキューからジョブをデキューするときに評価されるマクロを使用して動的に設定できます。次のマクロを設定できます。

| Macro             | Description                                           |
|-------------------|-------------------------------------------------------|
| `${project_name}` | run が ローンチ されている プロジェクト の名前。                     |
| `${entity_name}`  | run が ローンチ されている プロジェクト の所有者。                    |
| `${run_id}`       | ローンチ されている run の ID。                                  |
| `${run_name}`     | ローンチ されている run の名前。                                 |
| `${image_uri}`    | この run のコンテナ イメージの URI。                            |

{{% alert %}}
上記の表にリストされていないカスタムマクロ（`${MY_ENV_VAR}` など）は、 エージェント の 環境 から 環境 変数に置き換えられます。
{{% /alert %}}

## ローンチ エージェント を使用して、アクセラレータ（ GPU ）で実行されるイメージを構築する

アクセラレータ 環境 で実行されるイメージを構築するために ローンチ を使用する場合は、アクセラレータ ベース イメージを指定する必要がある場合があります。

このアクセラレータ ベース イメージは、次の要件を満たしている必要があります。

- Debian の互換性（ ローンチ Dockerfile は apt-get を使用して python をフェッチします）
- CPU と GPU のハードウェア命令セットの互換性（使用する予定の GPU で CUDA バージョンがサポートされていることを確認してください）
- 提供するアクセラレータ バージョンと ML アルゴリズムにインストールされているパッケージとの互換性
- ハードウェアとの互換性を設定するために追加の手順が必要なインストール済みパッケージ

### TensorFlow で GPU を使用する方法

TensorFlow が GPU を適切に利用していることを確認します。これを実現するには、キュー リソース 設定で `builder.accelerator.base_image` キーの Docker イメージとそのイメージ タグを指定します。

たとえば、`tensorflow/tensorflow:latest-gpu` ベース イメージは、TensorFlow が GPU を適切に使用することを保証します。これは、キュー内のリソース設定を使用して構成できます。

次の JSON スニペットは、キュー設定で TensorFlow ベース イメージを指定する方法を示しています。

```json title="Queue config"
{
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```