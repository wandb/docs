---
title: 利用情報と制限
description: W&B Inference の価格、利用制限、およびアカウント制限について理解する
linkTitle: Usage & Limits
weight: 20
---

W&B Inference を利用する前に、料金、利用制限、その他重要な利用情報についてご確認ください。

## 料金について

モデルごとの詳細な料金情報は [W&B Inference pricing](https://wandb.ai/site/pricing/inference) をご覧ください。

## クレジットの追加購入

W&B Inference クレジットは、Free、Pro、Academic プランに期間限定で付属しています。Enterprise プランの利用可否は異なる場合があります。クレジットがなくなった場合：

- **Free アカウント** は、W&B Inference の利用を継続するために有料プランへのアップグレードが必要です。[Pro または Enterprise へアップグレード](https://wandb.ai/subscriptions)
- **Pro プランのユーザー** は、[モデルごとの料金](https://wandb.ai/site/pricing/inference) に基づいて毎月、超過分が請求されます
- **Enterprise アカウント** は担当のアカウントエグゼクティブにお問い合わせください

## アカウント階層とデフォルト利用上限

各アカウント階層には、コスト管理や予期せぬ請求を防ぐためのデフォルトの利用上限が設定されています。W&B Inference の有料利用には事前の支払いが必要です。

利用上限の変更が必要な場合は、アカウントエグゼクティブまたはサポートまでご連絡ください。

| アカウント階層    | デフォルト上限    | 上限変更方法                                  |
|-------------------|------------------|--------------------------------------------|
| Pro               | $6,000/月         | アカウントエグゼクティブまたはサポートに連絡し、手動審査を依頼してください |
| Enterprise        | $700,000/年       | アカウントエグゼクティブまたはサポートに連絡し、手動審査を依頼してください |

## 同時実行数の制限

レートリミットを超えると、API は `429 Concurrency limit reached for requests` というレスポンスを返します。このエラーを解消するには、同時リクエスト数を減らしてください。詳細なトラブルシューティングは [W&B Inference サポート記事](/support/inference/) をご参照ください。

W&B では、W&B Project ごとにレートリミットが適用されます。たとえば、1つの Team に3つ Project がある場合、各 Project に独自のレートリミット枠が設けられています。

## Personal entity の非対応について

{{< alert title="Note" >}}
Personal entity は 2024年5月に廃止されたため、これはレガシーアカウントにのみ該当します。
{{< /alert >}}

Personal アカウント（personal entity）は W&B Inference をサポートしていません。W&B Inference を利用するには、Team の作成などでpersonalではないアカウントに切り替えてください。

## 地理的な制限

Inference サービスは、サポートされている地理的な地域からのみ利用できます。詳細は [Terms of Service](https://docs.coreweave.com/docs/policies/terms-of-service/terms-of-use#geographic-restrictions) をご参照ください。

## 次のステップ

- 利用開始前に [前提条件]({{< relref "prerequisites" >}}) をご確認ください
- [利用可能なモデル]({{< relref "models" >}}) とそれぞれのコストをご確認ください