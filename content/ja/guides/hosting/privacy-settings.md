---
title: プライバシー設定を行う
menu:
  default:
    identifier: ja-guides-hosting-privacy-settings
    parent: w-b-platform
weight: 4
---

組織とチームの管理者は、それぞれ組織スコープおよびチームスコープでプライバシー設定を設定できます。組織スコープで設定すると、その組織内のすべてのチームに対し組織管理者がその設定を適用します。

{{% alert %}}
W&B では、組織管理者がプライバシー設定を強制する場合は、事前にチーム管理者やユーザー全員にその旨を伝えることを推奨しています。これは思いがけないワークフローの変更を避けるためです。
{{% /alert %}}

## チームのプライバシー設定を行う

チーム管理者は、それぞれのチームの **Settings** タブ内の `Privacy` セクションからプライバシー設定を行うことができます。各設定は、組織スコープで強制されていない限り、個別に設定可能です。

* チームをすべての非メンバーから非表示にする
* 以後作成されるチームのすべてのプロジェクトをプライベートにする（一般公開を許可しない）
* 管理者以外のチームメンバーによるメンバー招待を許可する
* プライベートプロジェクト内の Reports のチーム外共有を無効にする。これにより既存のマジックリンクも無効化されます。
* 組織のメールアドレスドメインが一致するユーザーによるチーム参加を許可する
    * この設定は [SaaS Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ja" >}}) にのみ適用されます。[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) および [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}}) インスタンスでは利用できません。
* デフォルトでコード保存を有効にする

## すべてのチームにプライバシー設定を強制する

組織管理者は、アカウントや組織のダッシュボード内 **Settings** タブの `Privacy` セクションから、組織内すべてのチームにプライバシー設定を強制できます。組織管理者が設定を強制した場合、チーム管理者は各自のチームでその設定を変更できません。

* チームの公開範囲制限を強制する
    * このオプションを有効にすると、すべてのチームが非メンバーに対して非表示になります
* 以後のプロジェクトに対するプライバシーを強制する
    * このオプションを有効にすると、今後作成されるすべてのチームプロジェクトがプライベートまたは [restricted]({{< relref path="./iam/access-management/restricted-projects.md" lang="ja" >}}) で作成されます
* 招待操作の制御を強制する
    * このオプションを有効にすると、管理者以外によるメンバー招待ができなくなります
* Reports 共有の制御を強制する
    * このオプションを有効にすると、プライベートプロジェクト内の Reports の外部公開が無効化され、既存のマジックリンクも無効化されます
* チームへのセルフ参加制限を強制する
    * このオプションを有効にすると、組織のメールドメインが一致するユーザーによるセルフ参加を制限します
    * この設定は [SaaS Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ja" >}}) でのみ利用可能です。[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) および [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}}) では利用できません。
* デフォルトのコード保存制限を強制する
    * このオプションを有効にすると、すべてのチームでデフォルトでコード保存が無効化されます