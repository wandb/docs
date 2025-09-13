---
title: プライバシー設定を行う
menu:
  default:
    identifier: ja-guides-hosting-privacy-settings
    parent: w-b-platform
weight: 4
---

組織管理者とチーム管理者は、それぞれ組織スコープとチームスコープでプライバシー設定のセットを設定できます。組織スコープで設定された場合、組織管理者はその組織のすべてのチームに対してそれらの設定を強制します。

{{% alert %}}
W&B は、組織管理者がプライバシー設定を強制する際は、事前に組織内のすべてのチーム管理者と ユーザー に周知したうえで実施することを推奨します。これは、ワークフロー に予期しない変更が生じるのを避けるためです。
{{% /alert %}}

## チームのプライバシー設定を行う

チーム管理者は、チームの **Settings** タブ内にある `Privacy` セクションから、各チームのプライバシー設定を設定できます。組織スコープで強制されていない限り、各設定は変更可能です:

* このチームをメンバー以外のすべての ユーザー から非表示にする
* 今後作成されるチームの projects をすべて非公開にする（公開共有は不可）
* チームメンバー であれば誰でも他のメンバーを招待できるようにする（管理者のみではなく）
* 非公開 projects 内の Reports に対するチーム外への公開共有を無効にする。これにより既存のマジックリンクも無効になります。
* 組織のメールドメインが一致する ユーザー がこのチームに参加できるようにする。
    * この設定は [SaaS Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ja" >}}) にのみ適用されます。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) や [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}}) インスタンスでは利用できません。
* デフォルトで コード の保存を有効にする。

## すべてのチームにプライバシー設定を強制する

組織管理者は、アカウントまたは組織の ダッシュボード の **Settings** タブにある `Privacy` セクションから、組織内のすべてのチームに対してプライバシー設定を強制できます。組織管理者がある設定を強制すると、チーム管理者は自分のチーム内ではその設定を変更できなくなります。

* チームの表示制限を強制する
    * このオプションを有効にすると、すべてのチームをメンバー以外の ユーザー から非表示にします
* 将来の projects のプライバシーを強制する
    * このオプションを有効にすると、すべてのチームで今後作成される projects を非公開または [制限付き]({{< relref path="./iam/access-management/restricted-projects.md" lang="ja" >}}) に強制します
* 招待の管理を強制する
    * このオプションを有効にすると、管理者以外がどのチームにもメンバーを招待できないようにします
* Reports の共有制御を強制する
    * このオプションを有効にすると、非公開 projects 内の Reports の公開共有を無効にし、既存のマジックリンクを無効化します
* チームへの自己参加の制限を強制する
    * このオプションを有効にすると、組織のメールドメインが一致する ユーザー が任意のチームへ自己参加することを制限します
    * この設定は [SaaS Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ja" >}}) にのみ適用されます。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) や [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}}) インスタンスでは利用できません。
* コード保存のデフォルト制限を強制する
    * このオプションを有効にすると、すべてのチームでデフォルトの コード 保存をオフにします