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


## W&B の組織使用状況を表示する
組織のダッシュボードを使用して、組織に属するユーザーの全体像や、組織のユーザーが W&B をどのように使用しているか、次のようなプロパティを含めて視覚化できます。

* **Name**: ユーザーの名前とその W&B ユーザー名。
* **Last active**: ユーザーが最後に W&B を使用した時刻。これは、プロダクト内のページの閲覧、run のログをとったりその他のアクションを実行したり、ログインするなど、認証が必要なアクティビティを含みます。
* **Role**: ユーザーの役割。
* **Email**: ユーザーのメールアドレス。
* **Team**: ユーザーが所属するチームの名前。

### ユーザーのステータスを表示する
**Last Active** 列は、ユーザーが招待を保留中かアクティブなユーザーかを示します。ユーザーは 3 つの状態のいずれかです:

* **Invite pending**: 管理者が招待を送信しましたが、ユーザーが招待を受け入れていない状態。
* **Active**: ユーザーが招待を受け入れ、アカウントを作成した状態。
* **Deactivated**: 管理者がユーザーのアクセスを取り消した状態。

{{< img src="/images/hosting/view_status_of_user.png" alt="" >}}

### 組織が W&B をどのように利用しているかを表示・共有する
組織が W&B をどのように利用しているかを CSV 形式で表示します。

1. **Add user** ボタンの横にある三点リーダーを選択します。
2. ドロップダウンから **Export as CSV** を選択します。

    {{< img src="/images/hosting/export_org_usage.png" alt="" >}}

これにより、組織内のすべてのユーザーと、そのユーザーに関する詳細（ユーザー名、最後にアクティブだった時刻、役割、メールアドレスなど）を一覧表示する CSV ファイルがエクスポートされます。

### ユーザーのアクティビティを表示する
**Last Active** 列を使用して、個々のユーザーの **Activity summary** を取得します。

1. ユーザーの **Last Active** エントリの上にマウスを置きます。
2. ツールチップが表示され、ユーザーのアクティビティに関する情報の概要が提供されます。

{{< img src="/images/hosting/activity_tooltip.png" alt="" >}}

ユーザーが _active_ である条件は以下の通りです:
- W&B にログインします。
- W&B アプリで任意のページを表示します。
- run のログをとります。
- 実験を追跡するために SDK を使用します。
- W&B サーバーと何らかの形でインタラクトします。

### 時間経過に伴うアクティブユーザーを表示する
組織のダッシュボードにある **Users active over time** プロットを使用して、時間経過に伴うアクティブユーザーの集計概要を取得します（下の画像の最も右側のプロット）。

{{< img src="/images/hosting/dashboard_summary.png" alt="" >}}

ドロップダウンメニューを使用して、日、月、またはオールタイムに基づいて結果をフィルタリングできます。