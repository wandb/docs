---
title: プロジェクトのアクセス制御を管理する
description: 公開範囲スコープとプロジェクトレベルのロールを使ってプロジェクトへのアクセスを管理する
menu:
  default:
    identifier: ja-guides-hosting-iam-access-management-restricted-projects
    parent: access-management
---

W&B プロジェクトのスコープを定義することで、誰がそのプロジェクトを閲覧・編集・および W&B Runs を送信できるかを制限できます。

W&B チーム内の任意のプロジェクトのアクセス レベルを設定するには、いくつかの管理手段を組み合わせて使うことができます。**Visibility scope（公開範囲）** が、より上位の管理手段です。これにより、どのユーザーグループがプロジェクトを閲覧・または runs を送信できるかを管理できます。_Team_ または _Restricted_ の visibility scope を利用する場合は、**Project level roles** を使って各ユーザーごとのアクセス レベルを詳細に設定できます。

{{% alert %}}
プロジェクトのオーナー、チーム管理者、または組織管理者は、プロジェクトの visibility（公開範囲）を設定・編集できます。
{{% /alert %}}

## Visibility scopes（公開範囲）

プロジェクトの visibility scope（公開範囲）は、全部で 4 種類から選べます。公開性が高い順から、次の通りです。


| Scope | 説明 |
| ----- | ----- |
| Open | プロジェクトについて知っている人なら誰でも閲覧と runs または Reports の送信ができます。|
| Public | プロジェクトについて知っている人なら誰でも閲覧できます。runs や Reports を送信できるのは自分のチームだけです。|
| Team | 親チームのメンバーだけがプロジェクトを閲覧 & runs または Reports を送信できます。チーム外の人はプロジェクトにアクセスできません。|
| Restricted | 親チームから招待されたメンバーのみプロジェクトを閲覧 & runs や Reports を送信できます。|

{{% alert %}}
機密データや秘匿性のあるデータに関連したワークフローで共同作業したい場合は、プロジェクトのスコープを **Restricted** に設定しましょう。チーム内で restricted プロジェクトを作成すると、コラボレーションするために、そのプロジェクト専用のチームメンバーを個別に招待または追加できます（Experiments、Artifacts、Reports など）。

他のプロジェクトスコープと違い、Restricted プロジェクトにはチーム全員への暗黙的アクセス権が付与されません。一方で、チーム管理者は必要に応じて Restricted プロジェクトに参加できます。
{{% /alert %}}

### 新規または既存プロジェクトの公開範囲を設定する

プロジェクトを新規作成する時や、後で編集する際に、その visibility scope（公開範囲）を設定します。

{{% alert %}}
* プロジェクトの owner もしくはチーム管理者のみが visibility scope を設定・編集できます。
* チーム管理者がチーム設定で **Make all future team projects private (public sharing not allowed)** を有効化している場合、そのチーム上では **Open** および **Public** の公開範囲を利用できません。この場合、使用できるのは **Team** と **Restricted** のみです。
{{% /alert %}}

#### 新規プロジェクト作成時に visibility scope を設定する

1. SaaS Cloud、Dedicated Cloud、または Self-managed インスタンスの自分の W&B Organization へ移動します。
2. 左サイドバーの **My projects** セクションで **Create a new project** ボタンをクリックします。または、チームの **Projects** タブから右上の **Create new project** ボタンをクリックします。
3. 親チームを選択し、プロジェクト名を入力したあと、**Project Visibility** ドロップダウンから希望するスコープを選択します。
{{< img src="/images/hosting/restricted_project_add_new.gif" alt="Creating restricted project" >}}

**Restricted** を選択した場合は、下記の手順も行ってください。

4. **Invite team members** フィールドで、W&B チームメンバーの名前を 1 名以上入力してください。コラボレーションに必要なメンバーだけを追加しましょう。
{{< img src="/images/hosting/restricted_project_2.png" alt="Restricted project configuration" >}}

{{% alert %}}
Restricted プロジェクトのメンバーは、後から **Users** タブで追加・削除できます。
{{% /alert %}}

#### 既存プロジェクトの visibility scope を編集する

1. W&B Project へ移動します。
2. 左側カラムの **Overview** タブを選択します。
3. 右上の **Edit Project Details** ボタンをクリックします。
4. **Project Visibility** ドロップダウンから希望するスコープを選びます。
{{< img src="/images/hosting/restricted_project_edit.gif" alt="Editing restricted project settings" >}}

**Restricted** を選んだ場合、さらに次の手順を行ってください。

5. プロジェクトの **Users** タブに移動し、**Add user** ボタンを押して、特定のユーザーを Restricted プロジェクトへ招待します。

{{% alert color="secondary" %}}
* **Team** から **Restricted** へ visibility scope を変更した場合、必要なチームメンバーを招待しない限り、チームの全メンバーはそのプロジェクトへのアクセス権を失います。
* **Restricted** から **Team** へ visibility scope を変更した場合、チーム全メンバーがプロジェクトにアクセスできるようになります。
* Restricted プロジェクトのユーザーリストからチームメンバーを削除すると、その人はアクセス権を失います。
{{% /alert %}}

### Restricted scope に関するその他の注意事項

* Restricted プロジェクトでチーム レベルのサービスアカウントを利用したい場合は、そのサービスアカウントを個別に招待または追加する必要があります。デフォルトのままではアクセスできません。
* Restricted プロジェクトから他のプロジェクトへ run を移動することはできませんが、非 Restricted プロジェクトから Restricted プロジェクトへの移動は可能です。
* Restricted プロジェクトの公開範囲は、チーム設定で **Make all future team projects private (public sharing not allowed)** が有効でも、**Team** へ変更できます。
* Restricted プロジェクトの owner がもはや親チームのメンバーでない場合、プロジェクトの運用に支障が出ないよう、チーム管理者が owner を変更してください。

## Project level roles

チームの _Team_ または _Restricted_ スコープのプロジェクトでは、ユーザーごとに特定の role を割り当てられます。これはチームレベルの role とは異なる場合も可能です。例えば、あるユーザーがチームレベルで _Member_ だったとしても、そのチーム内の _Team_ や _Restricted_ スコープのプロジェクトで、そのユーザーに _View-Only_ や _Admin_、またはカスタム role を指定できます。

{{% alert %}}
Project level roles は SaaS Cloud、Dedicated Cloud、Self-managed インスタンス上でプレビュー提供中です。
{{% /alert %}}

### プロジェクトレベル role をユーザーに割り当てる

1. W&B Project へ移動します。
2. 左側カラムの **Overview** タブを選択します。
3. プロジェクトの **Users** タブを開きます。
4. **Project Role** 欄で、該当ユーザーの現在の role をクリックするとドロップダウンが開き、他の利用可能な role が表示されます。
5. ドロップダウンから別の role を選択します。変更内容は即時保存されます。

{{% alert %}}
ユーザーの project level role をチームレベルの role と異なるものに設定した場合、プロジェクトレベルの role 横に **\*** が表示され、差異が一目で分かるようになります。
{{% /alert %}}

### プロジェクトレベル role のその他の注意点

* デフォルトでは、_team_ または _restricted_ スコープのプロジェクト内全ユーザーの project level role は、それぞれのチームレベル role を **継承** します。
* チームレベルの role が _View-only_ のユーザーについては、project level role を変更 **できません**。
* project level role がチームレベル role と**同じ**場合、チーム管理者が後でチームレベル role を変更すると、関連する project role もそれに連動して自動的に変わります。
* project level role がチームレベル role と**異なる**場合、チーム管理者がチームレベル role を後で変更しても、該当ユーザーの project level role はそのまま維持されます。
* チームレベル role と project level role が異なるユーザーを _restricted_ プロジェクトから一度削除し、後日再追加した際は、デフォルト挙動によりチームレベル role が継承されます。再度 project level role を変更する必要があります。