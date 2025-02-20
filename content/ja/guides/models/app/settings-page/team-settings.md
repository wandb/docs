---
title: Manage team settings
description: チーム設定ページでチームメンバー、アバター、アラート、プライバシー設定を管理します。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-team-settings
    parent: settings
weight: 30
---

# Team settings

チームの設定を変更し、メンバー、アバター、アラート、プライバシー、使用状況を含めることができます。チームの設定を閲覧および編集できるのは、チーム管理者のみです。

{{% alert %}}
チーム設定の変更やチームからメンバーを削除できるのは、管理アカウントの種類のみです。
{{% /alert %}}


## Members
**Members** セクションには、すべての保留中の招待状と、チームへの招待を受け入れたメンバーの一覧が表示されます。各メンバーリストには、メンバーの名前、ユーザー名、メールアドレス、チーム役割、および Organization から受け継がれる Models と Weave へのアクセス権限が表示されます。標準的なチーム役割には、Administrator (Admin)、Member、および View-only の 3 つがあります。

チームの作成方法、ユーザーをチームへ招待する方法、ユーザーをチームから削除する方法、ユーザーの役割を変更する方法については、[Add and Manage teams]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}}) を参照してください。

## Avatar

**Avatar** セクションに移動し、画像をアップロードすることでアバターを設定できます。

1. **Update Avatar** を選択して、ファイルダイアログを表示させます。
2. ファイルダイアログから使用したい画像を選択します。

## Alerts

runs がクラッシュしたとき、終了したとき、またはカスタムアラートを設定したときにチームに通知します。チームはメールまたは Slack 経由でアラートを受け取ることができます。

受信したいイベントタイプの横にあるスイッチを切り替えます。Weights and Biases では、デフォルトで以下のイベントタイプオプションを提供しています：

* **Runs finished**: Weights and Biases の run が成功裏に終了したかどうか。
* **Run crashed**: run が終了できなかった場合。

アラートの設定方法や管理方法についての詳細は、[Send alerts with wandb.alert]({{< relref path="/guides/models/track/runs/alert.md" lang="ja" >}}) を参照してください。

## Privacy

プライバシー設定を変更するには、**Privacy** セクションへ移動します。プライバシー設定を変更できるのは、管理役割を持つメンバーのみです。管理者の役割は以下を実行できます：

* チーム内の Projects を非公開に強制する。
* デフォルトでコードの保存を有効にする。

## Usage

**Usage** セクションでは、Weights and Biases のサーバーでチームが消費したメモリの総使用量が記載されています。デフォルトのストレージプランは 100GB です。ストレージと価格についての詳細は、[Pricing](https://wandb.ai/site/pricing) ページをご覧ください。

## Storage

**Storage** セクションでは、チームのデータに使用されているクラウドストレージバケット設定について説明しています。詳細は [Secure Storage Connector]({{< relref path="teams.md#secure-storage-connector" lang="ja" >}}) を参照してください。また、自分でホスティングしている場合は、[W&B Server]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) ドキュメントをチェックしてください。