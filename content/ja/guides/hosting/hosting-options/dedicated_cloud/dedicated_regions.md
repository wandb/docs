---
title: サポートされている専用クラウドリージョン
menu:
  default:
    identifier: dedicated_regions
    parent: dedicated-cloud
url: guides/hosting/hosting-options/dedicated_regions
weight: 1
---

AWS、GCP、Azure は、世界中の複数拠点でクラウド コンピューティング サービスを提供しています。グローバルリージョンは、データの所在やコンプライアンス、レイテンシー、コスト効率などに関する要件を満たすのに役立ちます。W&B は Dedicated Cloud 向けに多くのグローバルリージョンをサポートしています。

{{% alert %}}
ご希望の AWS、GCP、Azure のリージョンが下記に記載されていない場合は、W&B サポートまでご連絡ください。W&B が該当リージョンに Dedicated Cloud の要件を満たすためのサービスが揃っているか検証し、その結果に基づき優先的にサポートを検討いたします。
{{% /alert %}}

## サポートされている AWS リージョン

以下の表は、W&B が現在 Dedicated Cloud インスタンスでサポートしている [AWS リージョン](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Concepts.RegionsAndAvailabilityZones.html) を示しています。

| リージョン所在地         | リージョン名       |
|-------------------|----------------|
|米国東部 (オハイオ)         | us-east-2        |
|米国東部 (北バージニア)     | us-east-1        |
|米国西部 (北カリフォルニア) | us-west-1        |
|米国西部 (オレゴン)         | us-west-2        |
|カナダ (中部)               | ca-central-1     |
|ヨーロッパ (フランクフルト)  | eu-central-1     |
|ヨーロッパ (アイルランド)    | eu-west-1        |
|ヨーロッパ (ロンドン)        | eu-west-2        |
|ヨーロッパ (ミラノ)          | eu-south-1       |
|ヨーロッパ (ストックホルム)  | eu-north-1       |
|アジアパシフィック (ムンバイ)| ap-south-1       |
|アジアパシフィック (シンガポール) | ap-southeast-1|
|アジアパシフィック (シドニー)     | ap-southeast-2|
|アジアパシフィック (東京)         | ap-northeast-1 |
|アジアパシフィック (ソウル)       | ap-northeast-2 |

AWS リージョンの詳細については、AWS ドキュメントの [リージョン、アベイラビリティーゾーン、ローカルゾーン](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Concepts.RegionsAndAvailabilityZones.html) をご覧ください。

AWS リージョン選択時に考慮すべきポイントの概要は、[What to Consider when Selecting a Region for your Workloads](https://aws.amazon.com/blogs/architecture/what-to-consider-when-selecting-a-region-for-your-workloads/) を参照してください。 

## サポートされている GCP リージョン

以下の表は、W&B が現在 Dedicated Cloud インスタンスでサポートしている [GCP リージョン](https://cloud.google.com/compute/docs/regions-zones) を示しています。

| リージョン所在地 | リージョン名             |
|------------------|--------------------------|
|サウスカロライナ     | us-east1                 |
|北バージニア        | us-east4                 |
|アイオワ           | us-central1              |
|オレゴン           | us-west1                 |
|ロサンゼルス       | us-west2                 |
|ラスベガス         | us-west4                 |
|トロント           | northamerica-northeast2  |
|ベルギー           | europe-west1             |
|ロンドン           | europe-west2             |
|フランクフルト      | europe-west3             |
|オランダ           | europe-west4             |
|シドニー           | australia-southeast1     |
|東京               | asia-northeast1          |
|ソウル             | asia-northeast3          |

GCP リージョンの詳細については、GCP ドキュメントの [リージョンとゾーン](https://cloud.google.com/compute/docs/regions-zones) をご覧ください。

## サポートされている Azure リージョン

以下の表は、W&B が現在 Dedicated Cloud インスタンスでサポートしている [Azure リージョン](https://azure.microsoft.com/explore/global-infrastructure/geographies/#geographies) を示しています。

| リージョン所在地   | リージョン名      |
|-------------------|------------------|
|バージニア         | eastus           |
|アイオワ           | centralus        |
|ワシントン         | westus2          |
|カリフォルニア     | westus           |
|カナダ中部         | canadacentral    |
|フランス中部       | francecentral    |
|オランダ           | westeurope       |
|東京・埼玉         | japaneast        |
|ソウル             | koreacentral     |

Azure リージョンの詳細については、Azure ドキュメントの [Azure 地理情報](https://azure.microsoft.com/explore/global-infrastructure/geographies/#overview) をご覧ください。