---
title: Projects の アクセス制御を管理する
description: 公開範囲スコープと Project レベルのロールで Project へのアクセスを管理する
menu:
  default:
    identifier: ja-guides-hosting-iam-access-management-restricted-projects
    parent: access-management
---

W&B Project のスコープを定義して、その Project を誰が閲覧・編集し、W&B Runs を送信できるかを制限します。
W&B の Team 内のあらゆる Project について、いくつかのコントロールを組み合わせて アクセス レベルを設定できます。**Visibility scope** は上位の仕組みで、どの Users グループがその Project を閲覧できるか、Runs を送信できるかを制御します。Project の公開範囲が _Team_ または _Restricted_ の 場合は、**Project level roles** を使って、各 User がその Project 内で持つ アクセス レベルを制御できます。

{{% alert %}}
Project のオーナー、Team 管理者、または組織の管理者が、その Project の公開範囲を設定・編集できます。
{{% /alert %}}

## Visibility scope（公開範囲）

選べる Project の Visibility scope は 4 種類あります。公開度が高い順に次のとおりです: 

| Scope | 説明 | 
| ----- | ----- |
| Open |Project について知っている人なら誰でも閲覧でき、Runs や Reports を送信できます。|
| Public |Project について知っている人なら誰でも閲覧できます。Runs や Reports を送信できるのはあなたの Team のみです。|
| Team | 親 Team のメンバーだけがその Project を閲覧し、Runs や Reports を送信できます。Team の外部の人はその Project に アクセス できません。 |
| Restricted| 親 Team から招待されたメンバーだけが、その Project を閲覧し、Runs や Reports を送信できます。|

{{% alert %}}
機微または機密の データ に関わるワークフローで共同作業する場合は、Project の scope を **Restricted** に設定してください。Team 内に Restricted Project を作成した場合、Team の特定メンバーだけを招待・追加して、関連する experiments、artifacts、reports などでコラボレーションできます。

他の scope と異なり、Team の全メンバーが自動的に Restricted Project へ アクセス できるわけではありません。一方で、必要に応じて Team 管理者は Restricted Project に参加できます。
{{% /alert %}}

### 新規または既存の Project に Visibility scope を設定する

Project を作成するとき、または後から編集するときに、その Project の Visibility scope を設定します。

{{% alert %}}
* Project のオーナーまたは Team 管理者のみが、その Visibility scope を設定・編集できます。
* Team のプライバシー設定で **Make all future team projects private (public sharing not allowed)** を有効にすると、その Team では **Open** と **Public** の Project Visibility scope が無効になります。この場合、使用できるのは **Team** と **Restricted** のみです。
{{% /alert %}}

#### 新しい Project を作成するときに Visibility scope を設定する

1. SaaS Cloud、Dedicated Cloud、または Self-managed インスタンス上の W&B の組織に移動します。
2. 左側サイドバーの **My projects** セクションで **Create a new project** ボタンをクリックします。あるいは、Team の **Projects** タブに移動し、右上の **Create new project** ボタンをクリックします。
3. 親 Team を選択し Project 名を入力したら、**Project Visibility** のドロップダウンから希望の scope を選びます。
{{< img src="/images/hosting/restricted_project_add_new.gif" alt="Restricted Project の作成" >}}

Visibility で **Restricted** を選んだ場合は、次の手順も実施します。

4. **Invite team members** フィールドに、1 人以上の W&B Team メンバーの名前を入力します。Project で共同作業が必要なメンバーのみを追加してください。
{{< img src="/images/hosting/restricted_project_2.png" alt="Restricted Project の設定" >}}

{{% alert %}}
Restricted Project では後から **Users** タブでメンバーを追加・削除できます。
{{% /alert %}}

#### 既存の Project の Visibility scope を編集する

1. 対象の W&B Project に移動します。
2. 左側のカラムで **Overview** タブを選択します。
3. 右上の **Edit Project Details** ボタンをクリックします。  
4. **Project Visibility** のドロップダウンから希望の scope を選びます。
{{< img src="/images/hosting/restricted_project_edit.gif" alt="Restricted Project の設定を編集" >}}

Visibility で **Restricted** を選んだ場合は、次の手順も実施します。

5. Project の **Users** タブに移動し、**Add user** ボタンをクリックして、特定の Users を Restricted Project に招待します。

{{% alert color="secondary" %}}
* Visibility scope を **Team** から **Restricted** に変更すると、必要な Team メンバーを Project に招待しない限り、その Team の全メンバーはその Project への アクセス を失います。
* Visibility scope を **Restricted** から **Team** に変更すると、その Team の全メンバーがその Project に アクセス できるようになります。
* Restricted Project の **Users** リストから Team メンバーを削除すると、そのメンバーはその Project への アクセス を失います。
{{% /alert %}}

### Restricted scope に関するその他のポイント

* Restricted Project で Team レベルのサービス アカウントを使う場合は、そのアカウントを Project に個別に招待または追加してください。そうしない限り、Team レベルのサービス アカウントは既定では Restricted Project に アクセス できません。
* Restricted Project から Runs を移動することはできませんが、非 Restricted の Project から Restricted Project へ Runs を移動することはできます。
* Team のプライバシー設定 **Make all future team projects private (public sharing not allowed)** に関わらず、Restricted Project の可視性は **Team** scope にのみ変更できます。
* Restricted Project のオーナーが親 Team に所属していない場合は、Project の円滑な運用のために Team 管理者がオーナーを変更してください。

## Project レベルのロール

Team 内の _Team_ または _Restricted_ scope の Project では、User に特定のロールを割り当てられます。これはその User の Team レベルのロールと異なる場合があります。たとえば、ある User が Team レベルで _Member_ ロールの場合でも、その Team の _Team_ または _Restricted_ scope の Project 内では、その User に _View-Only_ や _Admin_、その他の利用可能なカスタム ロールを割り当てられます。

{{% alert %}}
Project レベルのロールは、SaaS Cloud、Dedicated Cloud、および Self-managed インスタンスでプレビュー提供中です。
{{% /alert %}}

### User に Project レベルのロールを割り当てる

1. 対象の W&B Project に移動します。
2. 左側のカラムで **Overview** タブを選択します。
3. Project の **Users** タブに移動します。
4. 該当する User の **Project Role** 欄に表示されている現在のロールをクリックすると、他に選べるロールがドロップダウンで表示されます。
5. ドロップダウンから別のロールを選びます。選択は即座に保存されます。

{{% alert %}}
ある User の Project レベルのロールを Team レベルのロールと異なるものに変更すると、その Project レベルのロールには差異を示すために **\*** が付きます。
{{% /alert %}}

### Project レベルのロールに関するその他のポイント

* 既定では、 _team_ または _restricted_ scope の Project にいるすべての Users の Project レベルのロールは、それぞれの Team レベルのロールを継承します。
* Team レベルで _View-only_ ロールの User の Project レベルのロールは変更できません。
* 特定の Project 内で、ある User の Project レベルのロールが Team レベルのロールと同一であり、その後に Team 管理者が Team レベルのロールを変更した場合、該当する Project のロールは Team レベルのロールに追従して自動的に変更されます。
* 特定の Project 内で、ある User の Project レベルのロールを Team レベルのロールと異なるものに変更した場合、その後に Team 管理者が Team レベルのロールを変更しても、該当する Project レベルのロールはそのまま維持されます。
* ある User の Project レベルのロールが Team レベルのロールと異なる状態で、その User を _restricted_ Project から削除し、後で再度その Project に追加した場合、既定の挙動により Team レベルのロールが継承されます。必要に応じて、再度 Project レベルのロールを Team レベルと異なるものに変更してください。