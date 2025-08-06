---
title: 対応している専用クラウド地域
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-dedicated_cloud-dedicated_regions
    parent: dedicated-cloud
url: guides/hosting/hosting-options/dedicated_regions
weight: 1
---

AWS、GCP、Azure は世界中の複数のロケーションでクラウド コンピューティング サービスを提供しています。グローバルリージョンはデータの保存場所やコンプライアンス、レイテンシー、コスト効率などの要件を満たすために役立ちます。W&B は Dedicated Cloud 用に多くの利用可能なグローバルリージョンをサポートしています。

{{% alert %}}
ご希望の AWS、GCP、または Azure のリージョンがリストにない場合は、W&B サポートまでご連絡ください。W&B では、そのリージョンが Dedicated Cloud に必要なすべてのサービスに対応しているかどうかを確認し、評価の結果に応じてサポートの優先順位を決定します。
{{% /alert %}}

## サポートされている AWS リージョン

以下の表は、W&B が現在 Dedicated Cloud インスタンス向けにサポートしている [AWS リージョン](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Concepts.RegionsAndAvailabilityZones.html) を示しています。

| リージョン所在地 | リージョン名 |
|-------------|--------|
|US East (Ohio)| us-east-2|
|US East (N. Virginia)|us-east-1|
|US West (N. California)|us-west-1|
|US West (Oregon)|us-west-2|
|カナダ（セントラル）|ca-central-1|
|ヨーロッパ（フランクフルト）|eu-central-1|
|ヨーロッパ（アイルランド）|eu-west-1|
|ヨーロッパ（ロンドン）|eu-west-2|
|ヨーロッパ（ミラノ）|eu-south-1|
|ヨーロッパ（ストックホルム）|eu-north-1|
|アジアパシフィック（ムンバイ）|ap-south-1|
|アジアパシフィック（シンガポール）| ap-southeast-1|
|アジアパシフィック（シドニー）|ap-southeast-2|
|アジアパシフィック（東京）|ap-northeast-1|
|アジアパシフィック（ソウル）|ap-northeast-2|

AWS リージョンの詳細については、AWS ドキュメントの [リージョン、アベイラビリティゾーン、ローカルゾーン](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Concepts.RegionsAndAvailabilityZones.html) を参照してください。

AWS リージョン選択時の検討事項については、[ワークロードのリージョン選択時に考慮すべき事項](https://aws.amazon.com/blogs/architecture/what-to-consider-when-selecting-a-region-for-your-workloads/) をご覧ください。

## サポートされている GCP リージョン

以下の表は、W&B が現在 Dedicated Cloud インスタンス向けにサポートしている [GCP リージョン](https://cloud.google.com/compute/docs/regions-zones) を示しています。

| リージョン所在地 | リージョン名 |
|-------------|--------|
|サウスカロライナ|us-east1|
|N. バージニア|us-east4|
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

以下の表は、W&B が現在 Dedicated Cloud インスタンス向けにサポートしている [Azure リージョン](https://azure.microsoft.com/explore/global-infrastructure/geographies/#geographies) を示しています。

| リージョン所在地 | リージョン名 |
|-------------|--------|
|バージニア|eastus|
|アイオワ|centralus|
|ワシントン|westus2|
|カリフォルニア|westus|
|カナダ セントラル|canadacentral|
|フランス セントラル|francecentral|
|オランダ|westeurope|
|東京・埼玉|japaneast|
|ソウル|koreacentral|

Azure リージョンの詳細については、Azure ドキュメントの [Azure 地理](https://azure.microsoft.com/explore/global-infrastructure/geographies/#overview) を参照してください。