---
title: チーム設定を管理する
description: チーム設定ページでチームのメンバー、アバター、アラート、プライバシー設定を管理します。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-team-settings
    parent: settings
weight: 30
---

# チーム設定

チームの設定を変更します。メンバー、アバター、通知、プライバシー、利用状況を含みます。組織の管理者およびチームの管理者は、チームの設定を表示および編集できます。

{{% alert %}}
チーム設定を変更したり、チームからメンバーを削除できるのは、管理アカウントタイプのみです。
{{% /alert %}}

## メンバー
メンバーセクションでは、保留中の招待と、チームに参加する招待を受け入れたメンバーのリストを表示します。各メンバーのリストには、メンバーの名前、ユーザー名、メール、チームの役割、および Models や Weave へのアクセス権限が表示されます。これらは組織から継承されます。標準のチーム役割 **Admin**、**Member**、**View-only** から選択できます。組織が [カスタムロールの作成]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#create-custom-roles" lang="ja" >}})をしている場合、カスタムロールを割り当てることもできます。

チームの作成、管理、およびチームのメンバーシップと役割の管理についての詳細は、[Add and Manage Teams]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}}) を参照してください。新しいメンバーを招待できる人や、チームの他のプライバシー設定を設定するには、[プライバシー]({{< relref path="#privacy" lang="ja" >}}) を参照してください。

## アバター

**Avatar** セクションに移動して画像をアップロードすることで、アバターを設定します。

1. **Update Avatar** を選択し、ファイルダイアログを表示します。
2. ファイルダイアログから使用したい画像を選択します。

## アラート

run がクラッシュしたり、完了したり、カスタムアラートを設定したりしたときにチームに通知します。チームは、メールまたは Slack を通じてアラートを受け取ることができます。

受け取りたいイベントタイプの横にあるスイッチを切り替えます。Weights and Biases はデフォルトで以下のイベントタイプオプションを提供します:

* **Runs finished**: Weights and Biases の run が正常に完了したかどうか。
* **Run crashed**: run が完了できなかった場合。

アラートの設定と管理についての詳細は、[wandb.alert を使用したアラートの送信]({{< relref path="/guides/models/track/runs/alert.md" lang="ja" >}}) を参照してください。

## Slack 通知

Slack の送信先を設定し、チームの[オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}})が、新しいアーティファクトが作成されたときや、run のメトリックが設定された閾値に達したときなどに Registry やプロジェクトでイベントが発生すると通知を送信できるようにします。[Slack オートメーションの作成]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ja" >}})を参照してください。

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

## ウェブフック

チームの[オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}})が、新しいアーティファクトが作成されたときや、run のメトリックが設定された閾値に達したときなどに Registry やプロジェクトでイベントが発生すると動作するようにウェブフックを設定します。[Webhook オートメーションの作成]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ja" >}})を参照してください。

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

## プライバシー

**Privacy** セクションに移動してプライバシー設定を変更します。プライバシー設定を変更できるのは組織の管理者のみです。

- 今後のプロジェクトを公開したり、レポートを公開で共有したりする機能をオフにします。
- チームの管理者だけでなく、どのチームメンバーも他のメンバーを招待できます。
- デフォルトでコードの保存がオンになっているかどうかを管理します。

## 使用状況

**Usage** セクションでは、チームが Weights and Biases サーバーで消費した合計メモリ使用量について説明します。デフォルトのストレージプランは100GBです。ストレージと価格についての詳細は、[Pricing](https://wandb.ai/site/pricing) ページを参照してください。

## ストレージ

**Storage** セクションでは、チームのデータに対して使用されるクラウドストレージバケットの設定を説明します。詳細は [Secure Storage Connector]({{< relref path="teams.md#secure-storage-connector" lang="ja" >}}) を参照するか、セルフホスティングしている場合は [W&B Server]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) ドキュメントをチェックしてください。