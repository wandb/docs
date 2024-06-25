---
displayed_sidebar: default
---


# 高度なキューセットアップ
このページでは、追加のLaunchキューオプションを設定する方法について説明します。

## キュー設定テンプレートのセットアップ
Queue Config Templatesを使用して、計算消費のガードレールを管理します。メモリ消費量、GPU、実行時間などのフィールドに対してデフォルト値、最小値、および最大値を設定できます。

キューを設定テンプレートで設定した後、チームメンバーは定義した範囲内でのみフィールドを変更できます。

### キューテンプレートの設定
既存のキューにキューテンプレートを設定するか、新しいキューを作成することができます。

1. [https://wandb.ai/launch](https://wandb.ai/launch) のLaunchアプリに移動します。
2. テンプレートを追加したいキューの名前の横にある **View queue** を選択します。
3. **Config** タブを選択します。ここには、キューの作成時期、キュー設定、既存の起動時オーバーライドなどの情報が表示されます。
4. **Queue config** セクションに移動します。
5. テンプレートを作成したい設定のキーと値を特定します。
6. 設定の値をテンプレートフィールドに置き換えます。テンプレートフィールドは `{{variable-name}}` の形式を取ります。
7. **Parse configuration** ボタンをクリックします。設定を解析すると、W&Bは作成した各テンプレート用にキュー設定の下にタイルを自動的に作成します。
8. 生成された各タイルについて、キュー設定が許可するデータタイプ (文字列、整数、または浮動小数点数) を最初に指定する必要があります。**Type** ドロップダウンメニューからデータタイプを選択します。
9. データタイプに基づいて、各タイル内に表示されるフィールドを完成させます。
10. **Save config** をクリックします。

例えば、チームが使用できるAWSインスタンスを制限するテンプレートを作成したい場合、テンプレートフィールドを追加する前のキュー設定は次のようになります：

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

次に、**Parse configuration** をクリックします。**Queue config** の下に新しいタイル `aws-instance` が表示されます。

ここから **Type** ドロップダウンから文字列を選択します。これにより、ユーザーが選択できる値を指定するフィールドが表示されます。例えば、以下の画像では、チームの管理者がユーザーが選択できる2つの異なるAWSインスタンスタイプ (`ml.m4.xlarge` と `ml.p3.xlarge`) を設定しています：

![](/images/launch/aws_template_example.png)

## 動的にLaunchジョブを設定する
エージェントがキューからジョブをデキューする際に評価されるマクロを使用して、キュー設定を動的に設定できます。次のマクロを設定できます：

| マクロ             | 説明                                           |
|-------------------|---------------------------------------------------|
| `${project_name}` | Runが起動されるプロジェクトの名前。            |
| `${entity_name}`  | Runが起動されるプロジェクトの所有者。         |
| `${run_id}`       | 起動されるRunのID。                            |
| `${run_name}`     | 起動しているRunの名前。                        |
| `${image_uri}`    | このRunのコンテナイメージのURI。                |

:::info
上記のテーブルにリストされていないカスタムマクロ（例えば `${MY_ENV_VAR}`）は、エージェントの環境から環境変数で置き換えられます。
:::

## Launchエージェントを使用してアクセラレーター（GPU）で実行されるイメージを作成する
Launchを使用してアクセラレーター環境で実行されるイメージを作成する場合、アクセラレーターベースのイメージを指定する必要があるかもしれません。

このアクセラレータベースイメージは次の要件を満たす必要があります：

- Debian互換性（Launch Dockerfileはpythonを取得するためにapt-getを使用します）
- CPU & GPUハードウェア命令セットとの互換性（使用するCUDAバージョンが使用するGPUに対応していることを確認してください）
- 提供するアクセラレータバージョンとMLアルゴリズムにインストールされたパッケージ間の互換性
- ハードウェアと互換性を持つための追加セットアップが必要なパッケージがインストールされていること

### TensorFlowでGPUを使用する方法

TensorFlowがGPUを適切に利用することを確認します。これを達成するには、キューリソース設定の `builder.accelerator.base_image` キーにDockerイメージとそのイメージタグを指定します。

例えば、`tensorflow/tensorflow:latest-gpu` ベースイメージは、TensorFlowがGPUを適切に使用することを保証します。これはキュー設定のリソース設定を使用して設定できます。

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