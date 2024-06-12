---
displayed_sidebar: default
---

# Advanced queue set up
このページでは、追加のLaunchキューオプションの設定方法について説明します。

## キュー設定テンプレートのセットアップ
Queue Config Templatesを使用して、計算リソースの消費に対するガードレールを管理します。メモリ消費、GPU、実行時間などのフィールドに対してデフォルト値、最小値、および最大値を設定します。

設定テンプレートを使用してキューを設定した後、チームのメンバーは指定した範囲内でのみフィールドを変更できます。

### キューのテンプレートを設定する
既存のキューにテンプレートを設定するか、新しいキューを作成できます。

1. [https://wandb.ai/launch](https://wandb.ai/launch) のLaunchアプリに移動します。
2. テンプレートを追加したいキューの名前の横にある **View queue** を選択します。
3. **Config** タブを選択します。これにより、キューの作成日時、キュー設定、既存の起動時オーバーライドなどの情報が表示されます。
4. **Queue config** セクションに移動します。
5. テンプレートを作成したい設定キーと値を特定します。
6. 設定の値をテンプレートフィールドに置き換えます。テンプレートフィールドは `{{variable-name}}` の形式を取ります。
7. **Parse configuration** ボタンをクリックします。設定を解析すると、W&Bは自動的に各テンプレートのタイルをキュー設定の下に作成します。
8. 生成された各タイルについて、まずキュー設定が許可するデータ型（文字列、整数、または浮動小数点数）を指定する必要があります。これを行うには、**Type** ドロップダウンメニューからデータ型を選択します。
9. データ型に基づいて、各タイル内のフィールドを入力します。
10. **Save config** をクリックします。

例えば、チームが使用できるAWSインスタンスを制限するテンプレートを作成したいとします。テンプレートフィールドを追加する前のキュー設定は次のようになります：

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

`InstanceType` のテンプレートフィールドを追加すると、設定は次のようになります：

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

次に、**Parse configuration** をクリックします。**Queue config** の下に `aws-instance` とラベル付けされた新しいタイルが表示されます。

そこから、**Type** ドロップダウンからデータ型として文字列を選択します。これにより、ユーザーが選択できる値を指定するフィールドが表示されます。例えば、次の画像では、チームの管理者がユーザーが選択できる2つの異なるAWSインスタンスタイプ（`ml.m4.xlarge` と `ml.p3.xlarge`）を設定しています：

![](/images/launch/aws_template_example.png)

## Launchジョブを動的に設定する
キュー設定は、エージェントがキューからジョブをデキューするときに評価されるマクロを使用して動的に設定できます。次のマクロを設定できます：

| Macro             | 説明                                                   |
|-------------------|-------------------------------------------------------|
| `${project_name}` | runが起動されるプロジェクトの名前                      |
| `${entity_name}`  | runが起動されるプロジェクトの所有者                    |
| `${run_id}`       | 起動されるrunのID                                      |
| `${run_name}`     | 起動されるrunの名前                                    |
| `${image_uri}`    | このrunのコンテナイメージのURI                         |

:::info
前述の表に記載されていないカスタムマクロ（例：`${MY_ENV_VAR}`）は、エージェントの環境から環境変数で置き換えられます。
:::

## Launchエージェントを使用してアクセラレータ（GPU）で実行されるイメージをビルドする
アクセラレータ環境で実行されるイメージをビルドする場合、アクセラレータベースイメージを指定する必要があります。

このアクセラレータベースイメージは次の要件を満たす必要があります：

- Debian互換性（Launch Dockerfileはapt-getを使用してPythonを取得します）
- CPU & GPUハードウェア命令セットの互換性（使用するGPUがサポートするCUDAバージョンを確認してください）
- 提供するアクセラレータバージョンとMLアルゴリズムにインストールされているパッケージとの互換性
- ハードウェアとの互換性を設定するために追加の手順が必要なパッケージ

### TensorFlowでGPUを使用する方法

TensorFlowがGPUを適切に利用するようにします。これを達成するには、キューリソース設定の `builder.accelerator.base_image` キーにDockerイメージとそのイメージタグを指定します。

例えば、`tensorflow/tensorflow:latest-gpu` ベースイメージは、TensorFlowがGPUを適切に使用することを保証します。これはキューのリソース設定を使用して設定できます。

次のJSONスニペットは、キュー設定でTensorFlowベースイメージを指定する方法を示しています：

```json title="Queue config"
{
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```