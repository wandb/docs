---
title: レジストリへのアクセスを設定
menu:
  default:
    identifier: ja-guides-core-registry-configure_registry
    parent: registry
weight: 3
---

レジストリ管理者は、レジストリの設定から [レジストリロールの設定]({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}})、[ユーザーの追加]({{< relref path="configure_registry.md#add-a-user-or-a-team-to-a-registry" lang="ja" >}})、[ユーザーの削除]({{< relref path="configure_registry.md#remove-a-user-or-team-from-a-registry" lang="ja" >}}) を行うことができます。

## ユーザー管理

### ユーザーまたはチームの追加

レジストリ管理者は、個別のユーザーやチーム全体をレジストリに追加できます。ユーザーまたはチームを追加するには：

1. https://wandb.ai/registry/ にアクセスします。
2. ユーザーまたはチームを追加したいレジストリを選択します。
3. 画面右上の歯車アイコンをクリックしてレジストリ設定にアクセスします。
4. **Registry access** セクションで **Add access** をクリックします。
5. **Include users and teams** フィールドに、追加したいユーザー名、メールアドレス、またはチーム名を指定します。
6. **Add access** をクリックします。

{{< img src="/images/registry/add_team_registry.gif" alt="Adding teams to registry" >}}

[レジストリ内のユーザーロール設定についてさらに詳しく知る]({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}}) 、または [Registry のロール権限]({{< relref path="configure_registry.md#registry-role-permissions" lang="ja" >}}) も参照してください。

### ユーザーまたはチームの削除

レジストリ管理者は個別のユーザーやチーム全体をレジストリから削除できます。削除するには：

1. https://wandb.ai/registry/ にアクセスします。
2. ユーザーを削除したいレジストリを選択します。
3. 画面右上の歯車アイコンをクリックしてレジストリ設定にアクセスします。
4. **Registry access** セクションに移動し、削除したいユーザー名、メールアドレス、チーム名を入力します。
5. **Delete** ボタンをクリックします。

{{% alert %}}
ユーザーをチームから削除すると、そのユーザーのレジストリアクセスも削除されます。
{{% /alert %}}

## Registry ロール

各ユーザーには *registry ロール* が割り当てられ、そのレジストリで何ができるかが決まります。

W&B は、ユーザーやチームがレジストリに追加されると自動的にデフォルトロールを割り当てます。

| Entity | デフォルト registry ロール |
| ------ | ---------------------------- |
| Team | Viewer |
| User (管理者以外) | Viewer |
| Org admin | Admin |

レジストリ管理者は、レジストリ内のユーザーやチームにロールを割り当てたり変更したりできます。詳細は [レジストリ内のユーザーロールの設定]({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}}) をご覧ください。

{{% alert title="W&B のロールタイプ" %}}
W&B には2種類のロールがあります：[Team ロール]({{< ref "/guides/models/app/settings-page/teams.md#team-role-and-permissions" >}}) と [Registry ロール]({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}}) です。

チームでのロールは、レジストリでのロールに影響したり、関連したりしません。
{{% /alert %}}

以下の表は、ユーザーが持てるロールとその権限を示しています。

| 権限                                                     | 権限グループ | Viewer | Member | Admin | 
|-------------------------------------------------------|--------------|--------|--------|-------|
| コレクションの詳細を閲覧                              | Read         |   X    |   X    |   X   |
| リンクされたアーティファクトの詳細を閲覧               | Read         |   X    |   X    |   X   |
| 使用：use_artifact でアーティファクトを利用            | Read         |   X    |   X    |   X   |
| リンクされたアーティファクトをダウンロード             | Read         |   X    |   X    |   X   |
| アーティファクトのファイルビューワからファイルをDL     | Read         |   X    |   X    |   X   |
| レジストリを検索                                      | Read         |   X    |   X    |   X   |
| レジストリ設定やユーザーリストの閲覧                  | Read         |   X    |   X    |   X   |
| コレクション用自動化の新規作成                        | Create       |        |   X    |   X   |
| 新しいバージョン追加時にSlack通知を有効化             | Create       |        |   X    |   X   |
| 新規コレクションの作成                                | Create       |        |   X    |   X   |
| 新しいカスタムレジストリの作成                        | Create       |        |   X    |   X   |
| コレクションカード（説明文）の編集                     | Update       |        |   X    |   X   |
| リンクされたアーティファクト説明文の編集               | Update       |        |   X    |   X   |
| コレクションのタグ追加/削除                           | Update       |        |   X    |   X   |
| リンクされたアーティファクトのエイリアス追加/削除     | Update       |        |   X    |   X   |
| 新たなアーティファクトのリンク                         | Update       |        |   X    |   X   |
| レジストリの許可タイプリストを編集                     | Update       |        |   X    |   X   |
| カスタムレジストリ名の編集                            | Update       |        |   X    |   X   |
| コレクションの削除                                    | Delete       |        |   X    |   X   |
| 自動化フローの削除                                    | Delete       |        |   X    |   X   |
| レジストリからアーティファクトのリンク解除             | Delete       |        |   X    |   X   |
| レジストリの許可アーティファクトタイプの編集           | Admin        |        |        |   X   |
| レジストリ公開範囲の変更（Organization または Restricted）| Admin    |        |        |   X   |
| レジストリにユーザーを追加                            | Admin        |        |        |   X   |
| レジストリ内のユーザーロール割当・変更                 | Admin        |        |        |   X   |

### 継承される権限

レジストリ内のユーザーの権限は、そのユーザー個人またはチームメンバーシップによって割り当てられている最高レベルの権限に依存します。

例えば、レジストリ管理者が Nico というユーザーを Registry A に追加し、**Viewer** ロールを割り当てた場合。その後、管理者が Foundation Model Team というチームを Registry A に追加し、チームに **Member** ロールを割り当てたとします。

Nico は Foundation Model Team のメンバーなので、**Registry** でも **Member** になります。これは **Member** の権限が **Viewer** より高いため、W&B は Nico には **Member** ロールを適用します。

次の表は、個人ロールとチームロールが異なる場合の、最終的に適用される最高権限の例です。

| チーム registry ロール | 個人 registry ロール | 継承される registry ロール |
| ------- | -------------- | -------------------------- |
| Viewer  | Viewer         | Viewer                     |
| Member  | Viewer         | Member                     |
| Admin   | Viewer         | Admin                      | 

権限に競合がある場合、W&B はユーザー名の横に最高権限レベルを表示します。

例えば、下記画像では Alex は `smle-reg-team-1` チームに所属しているため **Member** 権限が継承されています。

{{< img src="/images/registry/role_conflict.png" alt="Registry role conflict resolution" >}}

## レジストリロールの設定

1. https://wandb.ai/registry/ にアクセスします。
2. 設定したいレジストリを選択します。
3. 画面右上の歯車アイコンをクリックします。
4. **Registry members and roles** セクションまでスクロールします。
5. **Member** フィールドで、権限を編集したいユーザーまたはチームを検索します。
6. **Registry role** 列で該当ユーザーのロールをクリックします。
7. ドロップダウンから割り当てたいロールを選択します。