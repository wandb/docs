---
title: プライバシー設定を 設定
menu:
  default:
    identifier: privacy-settings
    parent: w-b-platform
weight: 4
---

組織およびチームの管理者は、それぞれのスコープでプライバシー設定を設定できます。組織スコープで設定された場合、組織の管理者はその組織内のすべてのチームに対して設定を強制します。

{{% alert %}}
W&B では、組織管理者がプライバシー設定を強制する前に、その内容を組織内のすべてのチーム管理者やユーザーに事前に伝えることを推奨しています。これにより、ワークフローに予期せぬ影響が出るのを防ぐことができます。
{{% /alert %}}

## チームのプライバシー設定を変更する

チーム管理者は、自身のチームの **Settings** タブ内にある `Privacy` セクションからプライバシー設定を行えます。各設定は、組織スコープで強制されていなければ変更できます。

* チームを全メンバー以外から非表示にする
* 今後作成されるチームの Projects をすべて非公開に設定する（公開共有を許可しない）
* チーム内の任意のメンバーが他のメンバーを招待できるようにする（管理者のみでなくても可）
* 非公開 Projects の Reports の外部共有をオフにする（既存のマジックリンクも無効化されます）
* 組織のメールドメインと一致するユーザーがこのチームに参加できるようにする
    * この設定は [SaaS Cloud]({{< relref "./hosting-options/saas_cloud.md" >}}) のみ対象です。[Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud/" >}}) や [Self-managed]({{< relref "/guides/hosting/hosting-options/self-managed/" >}}) インスタンスでは利用できません。
* デフォルトでコード保存を有効にする

## すべてのチームに対してプライバシー設定を強制する

組織の管理者は、アカウントまたは組織のダッシュボード内 **Settings** タブの `Privacy` セクションから、組織内すべてのチームに対してプライバシー設定を強制できます。組織管理者が設定を強制すると、チーム管理者は各自のチームでその設定を変更できなくなります。

* チームの公開範囲制限を強制する
    * 有効にすると全チームが非メンバーから見えなくなります
* 今後の Projects のプライバシーを強制する
    * 有効にすると、組織内すべてのチームで今後作成される Projects が非公開または [restricted]({{< relref "./iam/access-management/restricted-projects.md" >}}) になります
* 招待コントロールを強制する
    * 有効にすると非管理者がメンバーをチームに招待できなくなります
* レポート共有コントロールを強制する
    * 有効にすると非公開 Projects 内の Reports の外部共有をオフにし、既存のマジックリンクも無効化されます
* チームの自己参加制限を強制する
    * 有効にすると、組織のメールドメインと一致するユーザーが任意のチームに自己参加することを制限します
    * この設定は [SaaS Cloud]({{< relref "./hosting-options/saas_cloud.md" >}}) のみ対象です。[Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud/" >}}) や [Self-managed]({{< relref "/guides/hosting/hosting-options/self-managed/" >}}) インスタンスでは利用できません。
* デフォルトのコード保存制限を強制する
    * 有効にするとすべてのチームでデフォルトのコード保存がオフになります