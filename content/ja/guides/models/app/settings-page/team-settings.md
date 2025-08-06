---
title: チーム設定を管理する
description: Team のメンバー、アバター、アラート、およびプライバシー設定を Team Settings ページで管理できます。
menu:
  default:
    identifier: team-settings
    parent: settings
weight: 30
---

# Team 設定

チームの設定（メンバー、アバター、アラート、プライバシー、使用状況など）を変更できます。Organization 管理者およびチーム管理者はチームの設定を閲覧・編集できます。

{{% alert %}}
チームの設定変更やメンバーの削除は、Administration アカウントタイプのみが可能です。
{{% /alert %}}


## Members
**Members** セクションでは、保留中の招待状と、招待を承諾してチームに参加しているメンバーの一覧が表示されます。各メンバーについては、名前、ユーザー名、メールアドレス、チームロール、および Organization から継承される Models や Weave へのアクセス権限が表示されます。標準のチームロールには **Admin**、**Member**、**View-only** があります。Organization で [カスタムロール]({{< relref "/guides/hosting/iam/access-management/manage-organization.md#create-custom-roles" >}}) を作成している場合は、カスタムロールを割り当てることもできます。

チームの作成・管理やメンバー・ロールの管理方法については、[Add and Manage teams]({{< relref "/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" >}}) をご参照ください。新しいメンバーの招待権限やチームの他のプライバシー設定を構成したい場合は、[Privacy]({{< relref "#privacy" >}}) を参照してください。

## Avatar

**Avatar** セクションに移動して画像をアップロードすることでアバターを設定できます。

1. **Update Avatar** を選択するとファイルダイアログが表示されます。
2. ファイルダイアログから使用したい画像を選択してください。

## Alerts

run がクラッシュしたり、完了したり、カスタムアラートを設定した時に、チームに通知が届くように設定できます。アラートはメールまたは Slack で受け取ることができます。

受け取りたいイベントタイプの横にあるスイッチを切り替えてください。Weights and Biases では、デフォルトで以下のイベントタイプが用意されています。

* **Runs finished**: Weights and Biases の run が正常に終了した場合。
* **Run crashed**: run が完了できず失敗した場合。

アラートの設定や管理方法の詳細は、[Send alerts with `wandb.Run.alert()`]({{< relref "/guides/models/track/runs/alert.md" >}}) をご参照ください。

## Slack notifications
Slack の送信先を設定し、Registry や Project でイベントが発生したとき（例：新しい artifact が作成された時や run のメトリクスがしきい値に達した時など）にチームの [automations]({{< relref "/guides/core/automations/" >}}) から通知を受け取ることができます。詳しくは [Create a Slack automation]({{< relref "/guides/core/automations/create-automations/slack.md" >}}) を参照してください。

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

## Webhooks
Webhook を設定して、Registry や Project でイベントが発生したとき（例：新しい artifact が作成された時や run のメトリクスがしきい値に達した時など）に、チームの [automations]({{< relref "/guides/core/automations/" >}}) が実行できるようにします。詳しくは [Create a webhook automation]({{< relref "/guides/core/automations/create-automations/slack.md" >}}) をご参照ください。

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

## Privacy

**Privacy** セクションでプライバシー設定を変更できます。プライバシー設定の変更は Organization 管理者のみ可能です。

- 今後の Project をパブリックにする権限や、Reports を公開で共有する機能を無効化できます。
- チーム管理者だけでなく、任意のチームメンバーが他のメンバーを招待できるように設定できます。
- コードの保存機能をデフォルトで有効にするかどうかを管理できます。

## Usage

**Usage** セクションでは、チームが Weights and Biases サーバー上で使用した総メモリ使用量が記載されています。デフォルトのストレージプランは100GBです。ストレージや料金についての詳細は [Pricing](https://wandb.ai/site/pricing) ページをご覧ください。

## Storage

**Storage** セクションでは、チームのデータを保存するクラウドストレージバケットの設定が記載されています。詳細は、[Secure Storage Connector]({{< relref "teams.md#secure-storage-connector" >}}) や、セルフホスティングの場合は [W&B Server]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) のドキュメントもご参照ください。