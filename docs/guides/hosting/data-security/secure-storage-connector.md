---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# BYOB（セキュアストレージコネクター）
BYOB（Bring your own bucket）は、W&B Artifactsやその他の機密データを自分のクラウドやオンプレインフラストラクチャーに保存することを可能にします。[Dedicated Cloud](../hosting-options/dedicated_cloud.md)または[SaaS Cloud](../hosting-options/saas_cloud.md)の場合、バケットに保存したデータはW&Bが管理するインフラストラクチャーにはコピーされません。

:::info
* W&B SDK / CLI / UIとバケット間の通信は[事前署名付きURL](./presigned-urls.md)を使用して行われます。
* W&Bはガベージコレクションプロセスを使用してW&B Artifactsを削除します。詳細は[Deleting Artifacts](../../artifacts/delete-artifacts.md)を参照してください。
:::

## 設定オプション
ストレージバケットの設定には、*インスタンスレベル*と*チームレベル*の2種類があります。

- インスタンスレベル: 組織内で関連する権限を持つユーザーは、インスタンスレベルのストレージバケットに保存されたファイルにアクセスできます。
- チームレベル: W&B Teamのメンバーは、そのチームレベルで設定されたバケットに保存されたファイルにアクセスできます。チームレベルのストレージバケットは、高機密データや厳格なコンプライアンス要件を持つチームにとって、より高度なデータアクセス制御とデータ分離を提供します。

インスタンスレベルと、組織内の1つまたは複数のチームに対して個別にバケットを設定することができます。

例えば、組織内にKappaというチームがあるとします。あなたの組織（およびチームKappa）はデフォルトでインスタンスレベルのストレージバケットを使用しています。次に、Omegaというチームを作成します。チームOmegaを作成する際には、チームレベルのストレージバケットを設定します。この場合、チームOmegaによって生成されたファイルはチームKappaによってアクセスされません。しかし、チームKappaによって作成されたファイルはチームOmegaによってアクセス可能です。チームKappaのデータを分離したい場合は、彼らのためにもチームレベルのストレージバケットを設定する必要があります。

:::tip
異なるビジネスユニットや部門がインスタンスを共有して効率的にインフラと管理リソースを利用する際、特にセルフマネージドインスタンスにおいて、チームレベルのストレージバケットは同様の利点を提供します。これは、異なる顧客案件に対してAIワークフローを管理するプロジェクトチームがある企業にも当てはまります。
:::

## 可用性マトリックス
以下の表は、異なるW&BサーバーデプロイメントタイプにおけるBYOBの可用性を示しています。`X`は特定のデプロイメントタイプでこの機能が使用可能であることを意味します。

| W&Bサーバーデプロイメントタイプ | インスタンスレベル | チームレベル | 追加情報 |
|----------------------------|--------------------|----------------|------------------------|
| Dedicated Cloud | X | X | インスタンスレベルとチームレベルのBYOBは、Amazon Web Services、Google Cloud Platform、Microsoft Azureで利用可能です。 |
| SaaS Cloud | | X | チームレベルのBYOBは、Amazon Web ServicesとGoogle Cloud Platformでのみ利用可能です。Microsoft Azureのデフォルトで唯一のストレージバケットはW&Bが完全に管理します。 |
| セルフマネージド | X | X | インスタンスレベルのBYOBがデフォルトですので、デプロイメントは完全にあなたが管理します。S3互換のセキュアストレージソリューション（例：[MinIO](https://github.com/minio/minio)）を使用することも可能です。 |

:::info
Dedicated CloudおよびMicrosoft Azure上のセルフマネージドインスタンスの場合、`SUPPORTED_FILE_STORES`という環境変数を使用してチームレベルで非Azureストレージバケットを使用することが可能です。詳細については、support@wandb.comでW&Bサポートにお問い合わせください。
:::

## ストレージオブジェクトの設定
ユースケースに基づいて、チームレベルまたはインスタンスレベルでストレージバケットを設定します。

:::info
システム管理者のみがストレージオブジェクトを設定する権限を持っています。
:::

:::tip
W&Bは、ストレージバケットと必要なIAM権限をプロビジョニングするために、[AWS](https://github.com/wandb/terraform-aws-wandb/tree/main/modules/secure_storage_connector)、[GCP](https://github.com/wandb/terraform-google-wandb/tree/main/modules/secure_storage_connector)、または[Azure](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector)で管理されているTerraformモジュールの使用を推奨します。
:::



<Tabs
  defaultValue="team"
  values={[
    {label: 'チームレベル', value: 'team'},
    {label: 'インスタンスレベル', value: 'instance'},
  ]}>
  <TabItem value="team">

W&B Teamを作成する際に、チームレベルでクラウドストレージバケットを設定します:

1. **Team Name**フィールドにチーム名を入力します。
2. **Company/Organization**ドロップダウンから、このチームが所属する会社または組織を選択します。  
3. **Choose storage type**オプションで**External Storage**を選択します。
4. ドロップダウンから**New bucket**を選択するか、既存のバケットを選択します。
:::tip
複数のW&B Teamsが同じクラウドストレージバケットを使用することができます。これを可能にするには、ドロップダウンから既存のクラウドストレージバケットを選択してください。
:::
5. **Cloud provider**ドロップダウンからクラウドプロバイダーを選択します。
6. **Name**フィールドにストレージオブジェクトの名前を入力します。
7. （AWSをご使用の場合は任意）**KMS key ARN**フィールドに暗号化キーのARNを入力します。
8. **Create Team**ボタンを選択します。

![](/images/hosting/prod_setup_secure_storage.png)

バケットへのアクセスに問題がある場合やバケットに無効な設定がある場合、ページの下部にエラーまたは警告が表示されます。


  </TabItem>
  <TabItem value="instance">

Dedicated CloudまたはセルフマネージドインスタンスのインスタンスレベルBYOBを設定するには、support@wandb.comでW&Bサポートにお問い合わせください。

  </TabItem>
</Tabs>