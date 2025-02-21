---
title: Configure privacy settings
menu:
  default:
    identifier: ja-guides-hosting-privacy-settings
    parent: w-b-platform
weight: 4
---

Organization と Team の管理者は、それぞれ Organization および Team のスコープで一連のプライバシー設定を構成できます。 Organization スコープで構成されている場合、Organization 管理者は、その Organization 内のすべての Team に対してこれらの設定を適用します。

{{% alert %}}
W&B は、Organization 管理者が、Organization 内のすべての Team 管理者と User に事前に通知した上で、プライバシー設定を適用することを推奨します。これは、ワークフローにおける予期しない変更を避けるためです。
{{% /alert %}}

## Team のプライバシー設定を構成する

Team 管理者は、Team の **Settings** タブの `Privacy` セクション内で、それぞれの Team のプライバシー設定を構成できます。各設定は、Organization スコープで適用されていない限り、構成可能です。

* この Team をすべての非メンバーから隠す
* 今後作成するすべての Team の Projects を非公開にする (パブリック共有は許可されません)
* すべての Team メンバーが他のメンバーを招待できるようにする (管理者だけでなく)
* プライベート Project 内の Reports の Team 外へのパブリック共有をオフにする。これにより、既存のマジックリンクが無効になります。
* 一致する Organization のメールアドレスドメインを持つ User がこの Team に参加できるようにする。
    * この設定は、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) にのみ適用されます。 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスでは使用できません。
* デフォルトで code の保存を有効にする。

## すべての Team に対してプライバシー設定を適用する

Organization 管理者は、アカウントまたは Organization のダッシュボードの **Settings** タブの `Privacy` セクション内で、Organization 内のすべての Team に対してプライバシー設定を適用できます。 Organization 管理者が設定を適用すると、Team 管理者はそれぞれの Team 内でその設定を構成できなくなります。

* Team の可視性制限を適用する
    * このオプションを有効にすると、すべての Team が非メンバーから隠されます。
* 今後作成する Projects のプライバシーを適用する
    * このオプションを有効にすると、すべての Team で今後作成されるすべての Projects が非公開または [restricted]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}}) になります。
* 招待管理を適用する
    * このオプションを有効にすると、管理者以外のメンバーが Team にメンバーを招待できなくなります。
* Report の共有管理を適用する
    * このオプションを有効にすると、プライベート Project 内の Reports のパブリック共有が無効になり、既存のマジックリンクが無効になります。
* Team への自己参加制限を適用する
    * このオプションを有効にすると、一致する Organization のメールアドレスドメインを持つ User が Team に自己参加できなくなります。
    * この設定は、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) にのみ適用されます。 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスでは使用できません。
* デフォルトの code 保存制限を適用する
    * このオプションを有効にすると、すべての Team でデフォルトで code の保存が無効になります。
