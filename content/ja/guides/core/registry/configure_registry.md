---
title: レジストリアクセスを設定する
menu:
  default:
    identifier: ja-guides-core-registry-configure_registry
    parent: registry
weight: 3
---

レジストリ管理者は、レジストリの設定を設定することで、[レジストリロールを設定]({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}})、[ユーザーを追加]({{< relref path="configure_registry.md#add-a-user-or-a-team-to-a-registry" lang="ja" >}})、または[ユーザーを削除]({{< relref path="configure_registry.md#remove-a-user-or-team-from-a-registry" lang="ja" >}})することができます。

## ユーザー管理

### ユーザーまたはチームの追加

レジストリ管理者は、個々のユーザーまたは全チームをレジストリに追加できます。ユーザーまたはチームをレジストリに追加するには、次の手順を実行します。

1. https://wandb.ai/registry/ に移動します。
2. ユーザーまたはチームを追加したいレジストリを選択します。
3. 右上隅のギアアイコンをクリックして、レジストリの設定にアクセスします。
4. **Registry access** セクションで **Add access** をクリックします。
5. **Include users and teams** フィールドに、追加したいユーザー名、メールアドレス、またはチーム名を指定します。
6. **Add access** をクリックします。

{{< img src="/images/registry/add_team_registry.gif" alt="UI を使用して個々のユーザーやチームをレジストリに追加するアニメーション" >}}

[レジストリでのユーザーロール設定]({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}})や[レジストリロールの権限]({{< relref path="configure_registry.md#registry-role-permissions" lang="ja" >}})についての詳細をご覧ください。

### ユーザーまたはチームの削除

レジストリ管理者は、個々のユーザーまたはチーム全体をレジストリから削除できます。ユーザーまたはチームをレジストリから削除するには、次の手順を実行します。

1. https://wandb.ai/registry/ に移動します。
2. ユーザーを削除したいレジストリを選択します。
3. 右上隅のギアアイコンをクリックして、レジストリの設定にアクセスします。
4. **Registry access** セクションに移動し、削除したいユーザー名、メールアドレス、またはチームを入力します。
5. **Delete** ボタンをクリックします。

{{% alert %}}
チームからユーザーを削除すると、そのユーザーのレジストリへのアクセスも削除されます。
{{% /alert %}}

## レジストリロール

レジストリ内の各ユーザーには *レジストリロール* があり、そのレジストリで何をできるかが決まります。

W&B は、レジストリにユーザーやチームが追加されると、自動的にデフォルトのレジストリロールを割り当てます。

| Entity | Default registry role |
| ----- | ----- |
| Team | Viewer |
| User (non admin) | Viewer |
| Org admin | Admin |

レジストリ管理者は、レジストリ内のユーザーやチームのロールを割り当てまたは変更することができます。詳細は [レジストリでのユーザーロールの設定]({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}}) を参照してください。

{{% alert title="W&Bロールタイプ" %}}
W&B には、[チームロール]({{< ref "/guides/models/app/settings-page/teams.md#team-role-and-permissions" >}})と[レジストリロール]({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}})の2種類のロールがあります。

チームにおけるあなたのロールは、いかなるレジストリにおけるあなたのロールにも影響や関連を持ちません。
{{% /alert %}}

以下の表は、ユーザーが持つことのできる異なるロールとその権限を示しています：

| Permission                                                     | Permission Group | Viewer | Member | Admin | 
|--------------------------------------------------------------- |------------------|--------|--------|-------|
| コレクションの詳細を表示する                                    | Read             |   X    |   X    |   X   |
| リンクされたアーティファクトの詳細を表示する                   | Read             |   X    |   X    |   X   |
| 使用: レジストリ内で use_artifact を使用してアーティファクトを使用 | Read             |   X    |   X    |   X   |
| リンクされたアーティファクトをダウンロードする                 | Read             |   X    |   X    |   X   |
| アーティファクトのファイルビューワーからファイルをダウンロードする | Read             |   X    |   X    |   X   |
| レジストリを検索する                                           | Read             |   X    |   X    |   X   |
| レジストリの設定とユーザーリストを表示する                     | Read             |   X    |   X    |   X   |
| コレクションの新しい自動化を作成する                           | Create           |        |   X    |   X   |
| 新しいバージョンが追加されたときの Slack 通知をオンにする      | Create           |        |   X    |   X   |
| 新しいコレクションを作成する                                   | Create           |        |   X    |   X   |
| 新しいカスタムレジストリを作成する                             | Create           |        |   X    |   X   |
| コレクションカード (説明) を編集する                           | Update           |        |   X    |   X   |
| リンクされたアーティファクトの説明を編集する                   | Update           |        |   X    |   X   |
| コレクションのタグを追加または削除する                         | Update           |        |   X    |   X   |
| リンクされたアーティファクトからエイリアスを追加または削除する | Update           |        |   X    |   X   |
| 新しいアーティファクトをリンクする                             | Update           |        |   X    |   X   |
| レジストリ用の許可されたタイプ一覧を編集する                   | Update           |        |   X    |   X   |
| カスタムレジストリ名を編集する                                 | Update           |        |   X    |   X   |
| コレクションを削除する                                         | Delete           |        |   X    |   X   |
| 自動化を削除する                                               | Delete           |        |   X    |   X   |
| レジストリからアーティファクトのリンクを解除する               | Delete           |        |   X    |   X   |
| レジストリ用の承認されたアーティファクトタイプを編集する       | Admin            |        |        |   X   |
| レジストリの公開範囲を変更する（組織または制限付き）           | Admin            |        |        |   X   |
| ユーザーをレジストリに追加する                                 | Admin            |        |        |   X   |
| レジストリ内のユーザーのロールを割り当てるまたは変更する       | Admin            |        |        |   X   |

### 継承された権限

レジストリ内のユーザーの権限は、そのユーザーに個別に、またはチームメンバーシップによって割り当てられた特権の最高レベルに依存します。

例えば、レジストリ管理者が Nico というユーザーをレジストリ A に追加し、**Viewer** レジストリロールを割り当てたとします。次に、レジストリ管理者が Foundation Model Team というチームをレジストリ A に追加し、Foundation Model Team に **Member** レジストリロールを割り当てたとします。

Nico は Foundation Model Team のメンバーであり、このチームは Registry の **Member** です。**Member** の権限は **Viewer** よりも高いため、W&B は Nico に **Member** ロールを付与します。

以下の表は、ユーザーの個別レジストリロールと、彼らが所属するチームのレジストリロールの間で矛盾が生じた場合の最高レベルの権限を示しています：

| Team registry role | Individual registry role | Inherited registry role |
| ------ | ------ | ------ | 
| Viewer | Viewer | Viewer |
| Member | Viewer | Member |
| Admin  | Viewer | Admin  | 

矛盾がある場合、W&B はユーザー名の横に最高レベルの権限を表示します。

例えば、以下の画像では、Alex は `smle-reg-team-1` チームのメンバーであるため、**Member** ロールの特権を継承しています。

{{< img src="/images/registry/role_conflict.png" alt="チームの一部であるため、メンバーのロールを継承するユーザー。" >}}

## レジストリロールの設定

1. https://wandb.ai/registry/ に移動します。
2. 設定したいレジストリを選択します。
3. 右上隅のギアアイコンをクリックします。
4. **Registry members and roles** セクションまでスクロールします。
5. **Member** フィールド内で、権限を編集したいユーザーまたはチームを検索します。
6. **Registry role** 列でユーザーのロールをクリックします。
7. ドロップダウンから、ユーザーに割り当てたいロールを選択します。