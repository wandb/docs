---
title: サポートされている専用クラウド リージョン
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-dedicated_cloud-dedicated_regions
    parent: dedicated-cloud
url: guides/hosting/hosting-options/dedicated_regions
weight: 1
---

AWS、GCP、Azure は、世界中の複数のロケーションでクラウドコンピューティングサービスを提供しています。グローバルリージョンは、データ所在地やコンプライアンス、レイテンシー、コスト効率といった要件の充足に役立ちます。W&B は、専用クラウドで利用可能な多くのグローバルリージョンをサポートしています。
{{% alert %}}
ご希望の AWS、GCP、または Azure リージョンがリストにない場合は、W&B サポートにお問い合わせください。W&B は、該当リージョンが専用クラウドに必要なすべてのサービスを備えているか検証し、評価結果に応じてサポートを優先します。
{{% /alert %}}

## サポートされている AWS リージョン
以下の表は、W&B が専用クラウド インスタンス向けに現在サポートしている [AWS リージョン](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Concepts.RegionsAndAvailabilityZones.html) を示します。

| リージョンロケーション | リージョン名 |
|-------------|--------|
|米国東部 (オハイオ)| us-east-2|
|米国東部 (バージニア北部)|us-east-1|
|米国西部 (北カリフォルニア)|us-west-1|
|米国西部 (オレゴン)|us-west-2|
|カナダ (中央)|ca-central-1|
|欧州 (フランクフルト)|eu-central-1|
|欧州 (アイルランド)|eu-west-1|
|欧州 (ロンドン)|eu-west-2|
|欧州 (ミラノ)|eu-south-1|
|欧州 (ストックホルム)|eu-north-1|
|アジア太平洋 (ムンバイ)|ap-south-1|
|アジア太平洋 (シンガポール)| ap-southeast-1|
|アジア太平洋 (シドニー)|ap-southeast-2|
|アジア太平洋 (東京)|ap-northeast-1|
|アジア太平洋 (ソウル)|ap-northeast-2|

AWS リージョンの詳細については、AWS ドキュメントの [リージョン、アベイラビリティーゾーン、ローカルゾーン](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Concepts.RegionsAndAvailabilityZones.html) を参照してください。

AWS リージョンを選択する際に考慮すべき要素の概要については、[ワークロードのリージョン選択時に考慮すべき事項](https://aws.amazon.com/blogs/architecture/what-to-consider-when-selecting-a-region-for-your-workloads/) を参照してください。

## サポートされている GCP リージョン
以下の表は、W&B が専用クラウド インスタンス向けに現在サポートしている [GCP リージョン](https://cloud.google.com/compute/docs/regions-zones) を示します。

| リージョンロケーション | リージョン名 |
|-------------|--------|
|サウスカロライナ|us-east1|
|バージニア北部|us-east4|
|アイオワ|us-central1|
|オレゴン|us-west1|
|ロサンゼルス|us-west2|
|ラスベガス|us-west4|
|トロント|northamerica-northeast2|
|ベルギー|europe-west1|
|ロンドン|europe-west2|
|フランクフルト|europe-west3|
|オランダ|europe-west4|
|シドニー|australia-southeast1|
|東京|asia-northeast1|
|ソウル|asia-northeast3|

GCP リージョンの詳細については、GCP ドキュメントの [リージョンとゾーン](https://cloud.google.com/compute/docs/regions-zones) を参照してください。

## サポートされている Azure リージョン
以下の表は、W&B が専用クラウド インスタンス向けに現在サポートしている [Azure リージョン](https://azure.microsoft.com/explore/global-infrastructure/geographies/#geographies) を示します。

| リージョンロケーション | リージョン名 |
|-------------|--------|
|バージニア|eastus|
|アイオワ|centralus|
|ワシントン|westus2|
|カリフォルニア|westus|
|カナダ中央|canadacentral|
|フランス中央|francecentral|
|オランダ|westeurope|
|東京、埼玉|japaneast|
|ソウル|koreacentral|

Azure リージョンの詳細については、Azure ドキュメントの [Azure の地域](https://azure.microsoft.com/explore/global-infrastructure/geographies/#overview) を参照してください。