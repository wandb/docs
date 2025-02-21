---
title: Configure launch queue
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-queue-advanced
    parent: set-up-launch
url: guides/launch/setup-queue-advanced
---

以下のページでは、 Launch キューオプションの設定方法について説明します。

## キュー設定テンプレートの設定
キュー設定テンプレートを使用して、コンピューティング消費量に対するガードレールを管理および管理します。メモリ消費量、 GPU 、ランタイム期間などのフィールドのデフォルト値、最小値、および最大値を設定します。

設定テンプレートを使用してキューを設定すると、チームのメンバーは、定義した範囲内でのみ、定義したフィールドを変更できます。

### キューテンプレートの設定
既存のキューでキューテンプレートを設定するか、新しいキューを作成できます。

1. [https://wandb.ai/launch](https://wandb.ai/launch) の Launch アプリケーションに移動します。
2. テンプレートを追加するキューの名前の横にある [ **キューの表示** ] を選択します。
3. [ **設定** ] タブを選択します。これにより、キューが作成された日時、キューの設定、既存の起動時のオーバーライドなど、キューに関する情報が表示されます。
4. [ **キュー設定** ] セクションに移動します。
5. テンプレートを作成する設定キーと 値 を特定します。
6. 設定内の 値 をテンプレートフィールドに置き換えます。テンプレートフィールドは、 `{{variable-name}}` の形式を取ります。
7. [ **設定の解析** ] ボタンをクリックします。設定を解析すると、作成した各テンプレートのキュー設定の下にタイルが自動的に作成されます。
8. 生成された各タイルについて、最初にキュー設定で許可される データ 型 (文字列、整数、または浮動小数点数) を指定する必要があります。これを行うには、[ **型** ] ドロップダウンメニューから データ 型を選択します。
9. データ 型に基づいて、各タイル内に表示されるフィールドに入力します。
10. [ **設定の保存** ] をクリックします。

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

次に、[ **設定の解析** ] をクリックします。 `aws-instance` というラベルの新しいタイルが [ **キュー設定** ] の下に表示されます。

そこから、[ **型** ] ドロップダウンからデータ型として [文字列] を選択します。これにより、ユーザーが選択できる 値 を指定できるフィールドが入力されます。たとえば、次の図では、チームの管理者が、ユーザーが選択できる 2 種類の AWS インスタンスタイプ ( `ml.m4.xlarge` と `ml.p3.xlarge` ) を設定しています。

{{< img src="/images/launch/aws_template_example.png" alt="" >}}

## Launch ジョブの動的な設定
キュー設定は、エージェント がキューからジョブをデキューするときに評価されるマクロを使用して動的に設定できます。次のマクロを設定できます。

| マクロ                   | 説明                                                                |
| --------------------- | ------------------------------------------------------------------- |
| `${project_name}`    | run の Launch 先の プロジェクト の名前。                                      |
| `${entity_name}`     | run の Launch 先の プロジェクト のオーナー。                                    |
| `${run_id}`          | Launch される run の ID。                                               |
| `${run_name}`        | Launch される run の名前。                                              |
| `${image_uri}`       | この run のコンテナイメージの URI。                                        |

{{% alert %}}
上記の表にリストされていないカスタムマクロ ( `${MY_ENV_VAR}` など) は、 エージェント の 環境 変数に置き換えられます。
{{% /alert %}}

## Launch エージェント を使用して、アクセラレータ (GPU) で実行されるイメージを構築する
Launch を使用してアクセラレータ環境で実行されるイメージを構築する場合は、アクセラレータベースイメージを指定する必要がある場合があります。

このアクセラレータベースイメージは、次の要件を満たしている必要があります。

- Debian 互換性 ( Launch Dockerfile は apt-get を使用して python をフェッチします)
- 互換性のある CPU と GPU ハードウェア命令セット (使用する予定の GPU で CUDA バージョンがサポートされていることを確認してください)
- 提供するアクセラレータ バージョンと ML アルゴリズムにインストールされているパッケージとの互換性
- ハードウェアとの互換性を設定するために追加の手順が必要なインストール済みパッケージ

### TensorFlow で GPU を使用する方法

TensorFlow が GPU を適切に利用していることを確認します。これを実現するには、キューリソース設定で `builder.accelerator.base_image` キーの Docker イメージとそのイメージタグを指定します。

たとえば、 `tensorflow/tensorflow:latest-gpu` ベースイメージを使用すると、TensorFlow が GPU を適切に使用できるようになります。これは、キューのリソース設定を使用して構成できます。

次の JSON スニペットは、TensorFlow ベースイメージをキュー設定で指定する方法を示しています。

```json title="Queue config"
{
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```