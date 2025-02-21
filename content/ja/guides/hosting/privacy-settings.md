---
title: Configure privacy settings
menu:
  default:
    identifier: ja-guides-hosting-privacy-settings
    parent: w-b-platform
weight: 4
---

組織およびチームの管理者は、それぞれのスコープでプライバシー設定を構成することができます。組織スコープで設定が行われた場合、組織の管理者はその設定をその組織内のすべてのチームに適用します。

{{% alert %}}
W&B は、組織の管理者に、すべてのチームの管理者およびユーザーに予め通知した後にのみプライバシー設定を適用することを推奨します。これは、ワークフローにおける予期しない変更を避けるためです。
{{% /alert %}}

## チームのプライバシー設定を構成する

チーム管理者は、それぞれのチームの **Settings** タブ内の `Privacy` セクションからプライバシー設定を構成できます。各設定は、組織スコープで強制されていない限り、構成可能です：

* チームをすべての非メンバーから隠す
* すべての将来のチームプロジェクトをプライベートにする（公開共有は許可されません）
* チームメンバーが他のメンバーを招待することを許可する（管理者だけでなく）
* プライベートプロジェクト内のレポートの外部共有をオフにする。これにより、既存のマジックリンクが無効になります。
* 組織のメールドメインに一致するユーザーがこのチームに参加することを許可する。
    * この設定は [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) にのみ適用されます。[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) や [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスでは利用できません。
* デフォルトでコードの保存を有効にする。

## すべてのチームに対してプライバシー設定を強制する

組織の管理者は、組織の **Settings** タブ内の `Privacy` セクションから、組織内のすべてのチームに対してプライバシー設定を強制することができます。組織の管理者が設定を強制すると、チーム管理者は各チーム内でその設定を構成することができません。

* チームの可視性制限を強制する
    * このオプションを有効にすると、すべての非メンバーからチームを隠すことができます
* 将来のプロジェクトに対するプライバシーを強制する
    * このオプションを有効にすると、すべてのチームの将来のプロジェクトをプライベートまたは [restricted]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}}) とすることができます
* 招待制御を強制する
    * このオプションを有効にすると、非管理者によるメンバーの招待を防止することができます
* レポート共有制御を強制する
    * このオプションを有効にすると、プライベートプロジェクト内でのレポートの公開共有をオフにし、既存のマジックリンクを無効にすることができます
* チームの自己参加制限を強制する
    * このオプションを有効にすると、組織のメールドメインに一致するユーザーが任意のチームに自己参加することを制限できます
    * この設定は [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) にのみ適用されます。[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) や [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスでは利用できません。
* デフォルトのコード保存制限を強制する
    * このオプションを有効にすると、すべてのチームでデフォルトでコード保存をオフにすることができます