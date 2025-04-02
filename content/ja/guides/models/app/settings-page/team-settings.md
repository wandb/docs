---
title: Manage team settings
description: Team Settings ページで、 Team のメンバー、アバター、アラート、およびプライバシー設定を管理します。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-team-settings
    parent: settings
weight: 30
---

# チーム設定

チームのメンバー、アバター、アラート、プライバシー、利用状況などの設定を変更します。Organization の管理者と チーム の管理者は、チーム の設定を表示および編集できます。

{{% alert %}}
管理アカウントタイプのみが、チーム設定を変更したり、チームからメンバーを削除したりできます。
{{% /alert %}}

## メンバー
「メンバー」セクションには、保留中の招待と、チームへの参加招待を承認したメンバーのリストが表示されます。リストに表示される各メンバーには、メンバーの名前、ユーザー名、メールアドレス、チームの役割、および Models と Weave へのアクセス権限が表示されます。これらは Organization から継承されます。標準のチームの役割 **Admin** 、 **Member** 、および **View-only** から選択できます。Organization が[カスタムロール]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#create-custom-roles" lang="ja" >}})を作成している場合は、代わりにカスタムロールを割り当てることができます。

チームの作成方法、チームの管理方法、チームのメンバーシップと役割の管理方法については、[チームの追加と管理]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}})を参照してください。誰が新しいメンバーを招待できるかを設定し、チームのその他のプライバシー設定を構成するには、[プライバシー]({{< relref path="#privacy" lang="ja" >}})を参照してください。

## アバター

**アバター**セクションに移動し、画像をアップロードしてアバターを設定します。

1. **アバターを更新**を選択して、ファイルダイアログを表示します。
2. ファイルダイアログから、使用する画像を選択します。

## アラート

run がクラッシュ、完了、またはカスタムアラートを設定したときに、チームに通知します。チームは、メールまたは Slack でアラートを受信できます。

アラートを受信するイベントタイプの横にあるスイッチを切り替えます。Weights and Biases は、デフォルトで次のイベントタイプのオプションを提供します。

* **Runs finished**: Weights and Biases の run が正常に完了したかどうか。
* **Run crashed**: run が完了しなかった場合。

アラートの設定と管理方法の詳細については、[wandb.alert でアラートを送信]({{< relref path="/guides/models/track/runs/alert.md" lang="ja" >}})を参照してください。

## Slack 通知
チームの [Automations]({{< relref path="/guides/core/automations/" lang="ja" >}}) が、新しい Artifact が作成されたときや、run メトリクスが定義されたしきい値を満たしたときなど、Registry または プロジェクト でイベントが発生したときに通知を送信できる Slack の送信先を設定します。[Slack オートメーションの作成]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ja" >}})を参照してください。

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

## Webhook
チームの [Automations]({{< relref path="/guides/core/automations/" lang="ja" >}}) が、新しい Artifact が作成されたときや、run メトリクスが定義されたしきい値を満たしたときなど、Registry または プロジェクト でイベントが発生したときに実行できる Webhook を設定します。[Webhook オートメーションの作成]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ja" >}})を参照してください。

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

## プライバシー

**プライバシー**セクションに移動して、プライバシー設定を変更します。プライバシー設定を変更できるのは、Organization の管理者のみです。

- 将来の プロジェクト を公開したり、 Reports を公開共有したりする機能をオフにします。
- チーム管理者だけでなく、チームメンバーが他のメンバーを招待できるようにします。
- コードの保存をデフォルトでオンにするかどうかを管理します。

## 利用状況

**利用状況**セクションでは、チームが Weights and Biases サーバーで使用した総メモリ使用量について説明します。デフォルトのストレージプランは 100GB です。ストレージと価格の詳細については、[価格](https://wandb.ai/site/pricing)ページを参照してください。

## ストレージ

**ストレージ**セクションでは、チームの データ に使用されている クラウド ストレージ バケットの設定について説明します。詳細については、[セキュアストレージコネクタ]({{< relref path="teams.md#secure-storage-connector" lang="ja" >}})を参照するか、セルフホスティングの場合は [W&B Server]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) のドキュメントを確認してください。
