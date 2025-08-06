---
title: 使用情報と制限
description: W&B Inference の料金、利用制限、およびアカウント制限について理解する
linkTitle: Usage & Limits
menu:
  default:
    identifier: ja-guides-inference-usage-limits
weight: 20
---

W&B Inference を利用する前に、料金、制限、その他の重要な利用情報についてご確認ください。

## 料金

詳細なモデルごとの料金情報は、[W&B Inference pricing](https://wandb.ai/site/pricing/inference) をご覧ください。

## クレジットの追加購入

W&B Inference クレジットは Free、Pro、Academic プランで期間限定で付与されます。Enterprise の利用可否は条件により異なります。クレジットがなくなった場合は以下の通りです。

- **Free アカウント**：継続利用には有料プランへのアップグレードが必要です。[Pro または Enterprise へアップグレード](https://wandb.ai/subscriptions)
- **Pro プラン ユーザー**：超過分は [モデルごとの料金](https://wandb.ai/site/pricing/inference) に基づき毎月請求されます
- **Enterprise アカウント**：アカウント担当者へご連絡ください

## アカウント種別ごとのデフォルト利用上限

各アカウント種別には、コスト管理や予期せぬ請求を防ぐためのデフォルト利用上限があります。W&B では、有料の Inference アクセスには事前支払いが必要です。

利用上限の変更が必要な場合は、アカウント担当者またはサポートまでご連絡ください。

| アカウント種別 | デフォルト上限 | 上限変更方法 |
|--------------|-------------|---------------------|
| Pro | $6,000/月 | アカウント担当者またはサポートに連絡し、手動で審査 |
| Enterprise | $700,000/年 | アカウント担当者またはサポートに連絡し、手動で審査 |

## 同時実行制限

レートリミットを超過すると、API から `429 Concurrency limit reached for requests` レスポンスが返されます。このエラーを解決するには、同時リクエスト数を減らしてください。詳しいトラブルシューティングについては、[W&B Inference サポート記事](/support/inference/) をご参照ください。

W&B では、各 W&B Project 単位でレートリミットを設定しています。たとえば、チーム内に 3 つの Project がある場合、それぞれの Project に別々のレートリミット枠が設けられています。

## Personal entities 非対応

{{< alert title="Note" >}}
W&B は 2024年 5月に personal entities を廃止しました。この内容はレガシーアカウントにのみ該当します。
{{< /alert >}}

パーソナルアカウント（personal entities）は W&B Inference を利用できません。利用するには Team を作成し、パーソナル以外のアカウントに切り替えてください。

## 地理的制限

Inference サービスは対応している地域のみ利用可能です。詳細は [Terms of Service](https://docs.coreweave.com/docs/policies/terms-of-service/terms-of-use#geographic-restrictions) をご確認ください。

## 次のステップ

- 利用を始める前に [前提条件]({{< relref path="prerequisites" lang="ja" >}}) をご確認ください
- [利用可能な models]({{< relref path="models" lang="ja" >}}) と各モデルの料金をご覧ください
