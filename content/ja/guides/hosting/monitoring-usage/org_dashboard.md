---
title: View organization dashboard
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-org_dashboard
    parent: monitoring-and-usage
---

{{% alert color="secondary" %}}
Organization dashboard は、[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) と [Self-managed instances]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) でのみ利用可能です。
{{% /alert %}}

## W&B の組織利用状況の表示
Organization dashboard を使用して、組織に所属する Users 、組織の Users による W&B の使用状況、および次のプロパティに関する全体像を把握します。

* **Name**: ユーザー名と W&B のユーザー名。
* **Last active**: ユーザーが最後に W&B を使用した時間。これには、プロダクト内のページの閲覧、Runs の ログ 、その他のアクションの実行、またはログインなど、認証を必要とするすべてのアクティビティが含まれます。
* **Role**: ユーザーのロール。
* **Email**: ユーザーのメールアドレス。
* **Team**: ユーザーが所属する Teams の名前。

### ユーザーのステータスの表示
**Last Active** 列には、ユーザーが招待保留中か、アクティブなユーザーかが表示されます。ユーザーのステータスは、次の 3 つのいずれかです。

* **Invite pending**: 管理者が招待を送信したが、ユーザーが招待を承諾していない。
* **Active**: ユーザーが招待を承諾し、アカウントを作成した。
* **Deactivated**: 管理者がユーザーのアクセスを取り消した。

{{< img src="/images/hosting/view_status_of_user.png" alt="" >}}

### 組織での W&B の使用状況の表示と共有
組織での W&B の使用状況を CSV 形式で表示します。

1. **Add user** ボタンの横にある 3 つのドットを選択します。
2. ドロップダウンから、**Export as CSV** を選択します。

    {{< img src="/images/hosting/export_org_usage.png" alt="" >}}

これにより、組織のすべての Users と、ユーザー名、最終アクティブのタイムスタンプ、ロール、メールなど、ユーザーに関する詳細が記載された CSV ファイルがエクスポートされます。

### ユーザーアクティビティの表示
**Last Active** 列を使用して、個々のユーザーの **Activity summary** を取得します。

1. ユーザーの **Last Active** エントリの上にマウスを置きます。
2. ツールチップが表示され、ユーザーのアクティビティに関する情報の概要が表示されます。

{{< img src="/images/hosting/activity_tooltip.png" alt="" >}}

ユーザーが _アクティブ_ なのは、次の場合です。
- W&B にログインする。
- W&B アプリで任意のページを表示する。
- Runs を ログ に記録する。
- SDK を使用して Experiments を追跡する。
- なんらかの方法で W&B サーバー を操作する。

### アクティブなユーザー数の経時的変化の表示
Organization dashboard の **Users active over time** プロットを使用して、アクティブなユーザー数の経時的変化の集計概要を取得します（下の画像の右端のプロット）。

{{< img src="/images/hosting/dashboard_summary.png" alt="" >}}

ドロップダウンメニューを使用して、日数、月数、またはすべての期間に基づいて result をフィルタリングできます。
