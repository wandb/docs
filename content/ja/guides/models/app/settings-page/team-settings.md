---
title: チーム設定を管理
description: チームのメンバー、アバター、アラート、およびプライバシー設定は、Team Settings ページで管理できます。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-team-settings
    parent: settings
weight: 30
---

# チーム設定

チームの設定（メンバー、アバター、アラート、プライバシー、利用状況など）を変更できます。組織管理者およびチーム管理者はチームの設定を表示・編集できます。

{{% alert %}}
管理者アカウントのみ、チーム設定の変更やチームメンバーの削除が可能です。
{{% /alert %}}


## メンバー
**Members** セクションでは、保留中の招待状と、招待を承諾してチームに参加しているメンバーの一覧が表示されます。各メンバーには、名前、ユーザー名、メールアドレス、チームロール、そして Models と Weave へのアクセス権が表示されます（これらの権限は Organization から継承されます）。標準のチームロール **Admin**, **Member**, **View-only** から選択できます。組織で[カスタムロール]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#create-custom-roles" lang="ja" >}})を作成している場合は、それを割り当てることもできます。

チームの作成や管理、チームメンバーシップやロールの管理方法については、[Add and Manage teams]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}})をご参照ください。チームへの新規メンバー招待の権限設定や、他のプライバシー設定の方法については、[Privacy]({{< relref path="#privacy" lang="ja" >}})をご覧ください。

## アバター

**Avatar** セクションから画像をアップロードしてアバターを設定できます。

1. **Update Avatar** を選択して、ファイルダイアログを開きます。
2. ダイアログから使用したい画像を選択してください。

## アラート

run のクラッシュ、完了、またはカスタムアラートの設定時にチームへ通知を送ることができます。アラートはメールか Slack で受け取れます。

受信したいイベントタイプの隣にあるスイッチを切り替えてアラートを設定します。Weights and Biases では、デフォルトで次のイベントタイプが用意されています。

* **Runs finished**: Weights and Biases の run が正常に完了した場合。
* **Run crashed**: run が失敗し、完了できなかった場合。

アラートの設定・管理方法についての詳細は、[Send alerts with `wandb.Run.alert()`]({{< relref path="/guides/models/track/runs/alert.md" lang="ja" >}})を参照してください。

## Slack 通知
Slack 宛先を設定すると、チームの [automations]({{< relref path="/guides/core/automations/" lang="ja" >}}) により、Registry やプロジェクトで新しい artifact 作成や run メトリクスが条件を満たした時など、イベントが発生した際に通知を送ることができます。[Create a Slack automation]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ja" >}}) をご覧ください。

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

## Webhooks
Webhook を設定すると、チームの [automations]({{< relref path="/guides/core/automations/" lang="ja" >}}) により、Registry やプロジェクトで新しい artifact が作成された時や run メトリクスが定義されたしきい値に達した時などに、Webhook を実行できます。[Create a webhook automation]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ja" >}}) を参照してください。

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

## プライバシー

**Privacy** セクションからプライバシー設定を変更できます。プライバシー設定の変更は組織管理者のみ可能です。

- 今後作成する Projects の一般公開や、Reports の公開共有機能を無効にします。
- 全チームメンバーからメンバー招待を許可するか、チーム管理者のみ許可するかを切り替えます。
- コード保存をデフォルトで有効にするかどうかを管理します。

## 利用状況

**Usage** セクションでは、Weights and Biases サーバーでチームがこれまでに消費した総メモリ使用量を確認できます。デフォルトのストレージプランは 100GB です。ストレージや料金の詳細は [Pricing](https://wandb.ai/site/pricing) ページをご確認ください。

## ストレージ

**Storage** セクションでは、チームのデータに利用されているクラウドストレージバケットの設定内容について説明します。詳しくは [Secure Storage Connector]({{< relref path="teams.md#secure-storage-connector" lang="ja" >}}) またはセルフホスティングの場合は [W&B Server]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) のドキュメントも参照してください。