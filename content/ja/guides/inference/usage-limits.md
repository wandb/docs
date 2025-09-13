---
title: 利用情報と制限
description: W&B Inference の 料金、利用制限、アカウント制限について理解する
linkTitle: Usage & Limits
menu:
  default:
    identifier: ja-guides-inference-usage-limits
weight: 20
---

W&B Inference を利用する前に、料金、制限、その他の重要な利用情報をご確認ください。

## 料金

モデルの料金の詳細は [W&B Inference pricing](https://wandb.ai/site/pricing/inference) を参照してください。

## クレジットの追加購入

W&B Inference のクレジットは、期間限定で Free、Pro、Academic の各プランに付属します。Enterprise での提供状況は異なる場合があります。クレジットを使い切った場合は、次のとおりです:

- **Free アカウント** は W&B Inference を継続利用するには有料プランにアップグレードする必要があります。 [Pro または Enterprise にアップグレード](https://wandb.ai/subscriptions)
- **Pro プランのユーザー** は、[モデルごとの料金](https://wandb.ai/site/pricing/inference) に基づき、超過分が毎月請求されます
- **Enterprise アカウント** はアカウント担当に連絡してください

## アカウント階層とデフォルトの利用上限

各アカウント階層には、コスト管理と予期せぬ請求の防止のためのデフォルトの支出上限が設定されています。有料の W&B Inference アクセスには前払いが必要です。

一部の ユーザー は上限の変更が必要な場合があります。上限の調整はアカウント担当またはサポートにご連絡ください。

| アカウント階層 | デフォルト上限 | 上限の変更方法 |
|--------------|-------------|---------------------|
| Pro | $6,000/month | 手動審査のためアカウント担当またはサポートに連絡 |
| Enterprise | $700,000/year | 手動審査のためアカウント担当またはサポートに連絡 |

## 同時実行の制限

レート制限を超えると、API は `429 Concurrency limit reached for requests` を返します。 このエラーを解消するには、同時リクエスト数を減らしてください。詳細なトラブルシューティングは [W&B Inference のサポート記事](/support/inference/) を参照してください。

W&B は W&B の各 Project ごとにレート制限を適用します。たとえば Team に Project が 3 つある場合、各 Project に独自のレート制限のクォータが設定されます。

## Personal entities は非対応

{{< alert title="注意" >}}
Personal entities は 2024 年 5 月に廃止されたため、これはレガシー アカウントのみに該当します。
{{< /alert >}}

個人アカウント（personal entities）では W&B Inference は利用できません。W&B Inference に アクセス するには、Team を作成して非個人アカウントに切り替えてください。

## 地理的な制限

W&B Inference サービスは、対応している地域からのみ利用できます。詳細は [Terms of Service](https://docs.coreweave.com/docs/policies/terms-of-service/terms-of-use#geographic-restrictions) を参照してください。

## 次のステップ

- 開始前に [前提条件]({{< relref path="prerequisites" lang="ja" >}}) を確認する
- [利用可能なモデル]({{< relref path="models" lang="ja" >}}) とそれぞれのコストを確認する