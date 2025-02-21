---
title: Manage team settings
description: Team Settings ページで、 Team のメンバー、アバター、アラート、プライバシー 設定 を管理できます。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-team-settings
    parent: settings
weight: 30
---

# Team の 設定

Team の設定を変更します。設定には、メンバー、アバター、アラート、プライバシー、および使用状況が含まれます。Team の設定を表示および編集できるのは、Team の管理者のみです。

{{% alert %}}
管理アカウントタイプのみが Team の設定を変更したり、Team からメンバーを削除したりできます。
{{% /alert %}}

## メンバー
「メンバー」セクションには、保留中の招待と、Team への参加招待を承諾したメンバーのリストが表示されます。リストに表示される各メンバーには、メンバーの名前、ユーザー名、メールアドレス、Team のロール、および Organization から継承された Models および Weave へのアクセス権が表示されます。標準的な Team のロールは、管理者 (Admin)、メンバー、および表示専用の 3 つです。

Team の作成、Team への User の招待、Team からの User の削除、User のロールの変更方法については、[Team の追加と管理]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}})を参照してください。

## アバター

**アバター**セクションに移動して画像をアップロードし、アバターを設定します。

1.  **アバターを更新** を選択して、ファイルダイアログを表示します。
2.  ファイルダイアログから、使用する画像を選択します。

## アラート

runs のクラッシュ時、完了時、またはカスタムアラートの設定時に、Team に通知します。Team は、メールまたは Slack でアラートを受信できます。

アラートを受信するイベントタイプの横にあるスイッチを切り替えます。Weights & Biases は、デフォルトで次のイベントタイプのオプションを提供します。

*   **Runs finished**: Weights & Biases の run が正常に完了したかどうか。
*   **Run crashed**: run が完了しなかった場合。

アラートの設定と管理の方法の詳細については、[wandb.alert でアラートを送信]({{< relref path="/guides/models/track/runs/alert.md" lang="ja" >}})を参照してください。

## プライバシー

**プライバシー**セクションに移動して、プライバシー設定を変更します。管理者ロールを持つメンバーのみが、プライバシー設定を変更できます。管理者ロールは次のことができます。

*   Team 内の Projects を強制的に非公開にする。
*   デフォルトで code の保存を有効にする。

## 使用状況

**使用状況**セクションでは、Team が Weights & Biases サーバーで使用した合計メモリ使用量を説明します。デフォルトのストレージプランは 100GB です。ストレージと料金の詳細については、[料金](https://wandb.ai/site/pricing)ページを参照してください。

## ストレージ

**ストレージ**セクションでは、Team の data に使用されている cloud ストレージ bucket の configuration について説明します。詳細については、[セキュアストレージコネクタ]({{< relref path="teams.md#secure-storage-connector" lang="ja" >}})を参照するか、セルフホスティングの場合は [W&B Server]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) のドキュメントを確認してください。
