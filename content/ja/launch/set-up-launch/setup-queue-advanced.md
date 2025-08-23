---
title: ローンチキューを設定
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-queue-advanced
    parent: set-up-launch
url: guides/launch/setup-queue-advanced
---

このページでは、ローンチキューのオプション設定方法について説明します。

## キュー設定テンプレートのセットアップ
Queue Config Template を使って、計算リソース消費のガードレールを管理・運用できます。メモリ消費量、GPU、実行時間のデフォルト、最小値、最大値などを設定できます。

設定テンプレートをキューに設定すると、チームメンバーは指定範囲内でのみあなたが定義したフィールドを変更できます。

### キューテンプレートの設定
既存のキューでキューテンプレートを設定することも、新しいキューを作成して設定することも可能です。

1. [W&B Launch App](https://wandb.ai/launch) にアクセスします。
2. テンプレートを追加したいキューの名前の横にある **View queue** を選択します。
3. **Config** タブを選択します。ここでは、キュー作成日時やキューの設定内容、既存のローンチ時の上書き設定などが表示されます。
4. **Queue config** セクションに移動します。
5. テンプレート化したい設定キーと値を特定します。
6. 設定の値をテンプレートフィールドに置き換えます。テンプレートフィールドは `{{variable-name}}` の形を取ります。
7. **Parse configuration** ボタンをクリックします。設定をパースすると、作成した各テンプレートごとにキュー設定の下にタイルが自動的に生成されます。
8. 生成された各タイルについて、まずキュー設定が許容するデータ型（string, integer, float のいずれか）を指定します。**Type** ドロップダウンからデータ型を選択します。
9. データ型に応じて、各タイルに表示されるフィールドを入力します。
10. **Save config** をクリックします。

たとえば、チームで使用可能な AWS インスタンスタイプを制限するテンプレートを作成したい場合、テンプレート追加前のキュー設定は次のようになっているかもしれません:

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

`InstanceType` にテンプレートフィールドを追加すると、設定は次のようになります:

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

続いて **Parse configuration** をクリックすると、`aws-instance` と表示された新しいタイルが **Queue config** の下に表示されます。

ここで、**Type** ドロップダウンから「String」を選択します。これにより、ユーザーが選択可能な値を指定できる入力欄が現れます。たとえば、下記画像のように、チーム管理者がユーザーが選べる2種類の AWS インスタンスタイプ（`ml.m4.xlarge` と `ml.p3.xlarge`）を設定できます。

{{< img src="/images/launch/aws_template_example.png" alt="AWS CloudFormation template" >}}

## ローンチジョブの動的設定
キュー設定では、ジョブがキューからエージェントでデキューされるタイミングで評価されるマクロを使って動的に設定可能です。利用可能なマクロは以下の通りです。

| マクロ              | 説明                                                         |
|--------------------|--------------------------------------------------------------|
| `${project_name}`  | Run をローンチするプロジェクト名                              |
| `${entity_name}`   | Run をローンチするプロジェクトのオーナー                      |
| `${run_id}`        | ローンチされる Run のID                                      |
| `${run_name}`      | ローンチされる Run の名前                                    |
| `${image_uri}`     | この Run 用コンテナイメージの URI                            |

{{% alert %}}
上記以外のカスタムマクロ（例えば `${MY_ENV_VAR}` など）は、エージェントの環境変数で値が置換されます。
{{% /alert %}}

## Launch agent でアクセラレータ（GPU）対応のイメージをビルドする
アクセラレータ環境で実行されるイメージをローンチでビルドする場合は、アクセラレータ用ベースイメージを指定する必要があります。

このベースイメージは次の条件を満たす必要があります。

- Debian 互換性（Launch の Dockerfile では apt-get で python を取得します）
- CPU & GPU ハードウェア命令セットとの互換性（使用予定の GPU で CUDA バージョンが対応しているか確認してください）
- 提供するアクセラレータのバージョンと ML アルゴリズムでインストールされるパッケージの互換性
- ハードウェアとの互換性に追加のセッティングが必要なパッケージのインストール

### TensorFlow で GPU を使用する方法

TensorFlow に GPU を正しく利用させるには、キューのリソース設定で `builder.accelerator.base_image` キーに使用する Docker イメージとそのタグを指定してください。

たとえば、`tensorflow/tensorflow:latest-gpu` のベースイメージを指定すると、TensorFlow が GPU を正しく利用します。これはキューのリソース設定で設定できます。

以下の JSON スニペットは、キュー設定で TensorFlow のベースイメージを指定する例です。

```json title="Queue config"
{
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```