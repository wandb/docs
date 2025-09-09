---
title: Team 設定を管理
description: Team Settings ページで、チームのメンバー、アバター、アラート、プライバシー設定を管理します。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-team-settings
    parent: settings
weight: 30
---

# Team の設定

メンバー、アバター、アラート、プライバシー、使用状況など、Team の設定を変更できます。組織の管理者と Team 管理者は、Team の設定を表示および編集できます。

{{% alert %}}
Team の設定を変更したり、Team からメンバーを削除できるのは、管理者アカウント種別のみです。
{{% /alert %}}

## メンバー
Members セクションには、保留中の招待と、招待を受け入れて Team に参加したメンバーの一覧が表示されます。各メンバーには、氏名、ユーザー名、メール、Team ロールに加えて、Organization から継承される Models と W&B Weave へのアクセス権限が表示されます。標準の Team ロールは **Admin**、**Member**、**View-only** です。組織で [カスタム ロール]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#create-custom-roles" lang="ja" >}}) を作成している場合は、カスタム ロールを割り当てられます。

Team の作成方法、Team の管理、メンバーシップやロールの管理については、[Teams を追加・管理する]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}}) を参照してください。新しいメンバーを誰が招待できるかや、その他のプライバシー設定を構成するには、[Privacy]({{< relref path="#privacy" lang="ja" >}}) を参照してください。

## アバター

**Avatar** セクションに移動して画像をアップロードし、アバターを設定します。

1. **Update Avatar** を選択すると、ファイル ダイアログが表示されます。
2. ファイル ダイアログで使用する画像を選択します。

## アラート

Runs がクラッシュまたは完了したとき、あるいはカスタム アラートを設定して、Team に通知できます。アラートはメールまたは Slack で受け取れます。

受け取りたいイベント タイプの横にあるスイッチを切り替えます。Weights and Biases は既定で次のイベント タイプを提供します:

* **Runs finished**: Weights and Biases の Run が正常に完了したかどうか。
* **Run crashed**: Run が完了せずにクラッシュした場合。

アラートの設定と管理の詳細は、[wandb.Run.alert() でアラートを送信]({{< relref path="/guides/models/track/runs/alert.md" lang="ja" >}}) を参照してください。

## Slack 通知
Registry や project でイベントが発生したとき (たとえば新しい artifact が作成された場合や、Run のメトリクスが定義済みのしきい値を満たした場合など) に、Team の [オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}}) が通知を送信する Slack 宛先を設定します。詳しくは [Slack オートメーションを作成]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ja" >}}) を参照してください。

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

## Webhooks
Registry や project でイベントが発生したとき (たとえば新しい artifact が作成された場合や、Run のメトリクスが定義済みのしきい値を満たした場合など) に、Team の [オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}}) が実行する Webhook を設定します。詳しくは [Webhook オートメーションを作成]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ja" >}}) を参照してください。

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

## プライバシー

**Privacy** セクションに移動してプライバシー設定を変更します。プライバシー設定を変更できるのは組織の管理者のみです。

- 今後 Projects を公開したり、Reports を一般公開で共有したりできないようにします。
- Team 管理者だけでなく、任意の Team メンバーが他のメンバーを招待できるようにします。
- コード保存を既定で有効にするかどうかを管理します。

## 使用状況

**Usage** セクションでは、Team が Weights and Biases のサーバー上で消費したメモリ総量が示されます。既定のストレージ プランは 100GB です。ストレージと料金の詳細は、[Pricing](https://wandb.ai/site/pricing) ページを参照してください。

## ストレージ

**Storage** セクションでは、Team のデータに使用されているクラウド ストレージ バケットの設定について説明します。詳しくは [Secure Storage Connector]({{< relref path="teams.md#secure-storage-connector" lang="ja" >}}) を参照するか、セルフホスティングしている場合は [W&B Server]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) のドキュメントをご覧ください。