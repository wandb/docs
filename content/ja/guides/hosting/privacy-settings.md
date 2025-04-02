---
title: Configure privacy settings
menu:
  default:
    identifier: ja-guides-hosting-privacy-settings
    parent: w-b-platform
weight: 4
---

組織と Team の管理者は、それぞれ組織と Team のスコープで一連のプライバシー設定を構成できます。組織スコープで構成した場合、組織管理者はその設定を組織内のすべての Team に対して適用します。

{{% alert %}}
W&B は、組織管理者が組織内のすべての Team 管理者と User に事前に伝えてから、プライバシー設定を適用することを推奨します。これは、ワークフローにおける予期しない変更を避けるためです。
{{% /alert %}}

## Team のプライバシー設定を構成する

Team 管理者は、Team の **Settings** タブの `Privacy` セクション内で、それぞれの Team のプライバシー設定を構成できます。各設定は、組織スコープで適用されていない限り構成可能です。

*   この Team をすべての非メンバーから隠す
*   今後のすべての Team の Projects をプライベートにする (パブリック共有は許可されません)
*   すべての Team メンバーが他のメンバーを招待できるようにする (管理者だけでなく)
*   プライベート Project の Reports について、Team の外部へのパブリック共有をオフにします。これにより、既存の Magic Link がオフになります。
*   組織の Email ドメインが一致する User がこの Team に参加できるようにする。
    *   この設定は [SaaS Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ja" >}}) にのみ適用されます。[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}}) インスタンスでは利用できません。
*   デフォルトで Code の保存を有効にする。

## すべての Team にプライバシー設定を適用する

組織管理者は、アカウントまたは組織の ダッシュボード の **Settings** タブの `Privacy` セクション内で、組織内のすべての Team にプライバシー設定を適用できます。組織管理者が設定を適用すると、Team 管理者はそれぞれの Team 内でそれを構成できなくなります。

*   Team の可視性制限を適用する
    *   このオプションを有効にすると、すべての Team が非メンバーから隠されます
*   今後の Projects にプライバシーを適用する
    *   このオプションを有効にすると、すべての Team の今後のすべての Projects がプライベートまたは [制限付き]({{< relref path="./iam/access-management/restricted-projects.md" lang="ja" >}}) になるように適用されます
*   招待コントロールを適用する
    *   このオプションを有効にすると、管理者以外のメンバーが Team にメンバーを招待できなくなります
*   Report の共有コントロールを適用する
    *   このオプションを有効にすると、プライベート Project 内の Reports のパブリック共有が無効になり、既存の Magic Link が無効になります
*   Team のセルフ参加制限を適用する
    *   このオプションを有効にすると、組織の Email ドメインが一致する User が Team にセルフ参加できなくなります
    *   この設定は [SaaS Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ja" >}}) にのみ適用されます。[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}}) インスタンスでは利用できません。
*   デフォルトの Code 保存制限を適用する
    *   このオプションを有効にすると、すべての Team でデフォルトで Code の保存が無効になります
