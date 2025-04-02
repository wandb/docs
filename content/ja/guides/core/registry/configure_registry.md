---
title: Configure registry access
menu:
  default:
    identifier: ja-guides-core-registry-configure_registry
    parent: registry
weight: 3
---

Registry の管理者は、Registry の設定を構成することで、[Registry のロールを設定]({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}})、[ユーザーを追加]({{< relref path="configure_registry.md#add-a-user-or-a-team-to-a-registry" lang="ja" >}})、または [ユーザーを削除]({{< relref path="configure_registry.md#remove-a-user-or-team-from-a-registry" lang="ja" >}}) できます。

## ユーザーの管理

### ユーザーまたは Teams の追加

Registry の管理者は、個々の ユーザー または Teams 全体を Registry に追加できます。 ユーザー または Teams を Registry に追加するには:

1. Registry (https://wandb.ai/registry/) に移動します。
2. ユーザー または Teams を追加する Registry を選択します。
3. 右上隅にある歯車アイコンをクリックして、Registry の 設定 にアクセスします。
4. **Registry access** セクションで、**Add access** をクリックします。
5. **Include users and teams** フィールドに、1 つ以上の ユーザー 名、メールアドレス、または Teams 名を指定します。
6. **Add access** をクリックします。

{{< img src="/images/registry/add_team_registry.gif" alt="Animation of using the UI to add teams and individual users to a registry" >}}

[Registry での ユーザー ロールの設定]({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}})、または [Registry ロールの権限]({{< relref path="configure_registry.md#registry-role-permissions" lang="ja" >}}) について詳細をご覧ください。

### ユーザー または Teams の削除
Registry の管理者は、個々の ユーザー または Teams 全体を Registry から削除できます。 ユーザー または Teams を Registry から削除するには:

1. Registry (https://wandb.ai/registry/) に移動します。
2. ユーザー を削除する Registry を選択します。
3. 右上隅にある歯車アイコンをクリックして、Registry の 設定 にアクセスします。
4. **Registry access** セクションに移動し、削除する ユーザー 名、メールアドレス、または Teams を入力します。
5. **Delete** ボタンをクリックします。

{{% alert %}}
Teams から ユーザー を削除すると、その ユーザー の Registry へのアクセス権も削除されます。
{{% /alert %}}

## Registry ロール

Registry 内の各 ユーザー は、その Registry で何ができるかを決定する _Registry ロール_ を持っています。

W&B は、ユーザー または Teams が Registry に追加されると、デフォルトの Registry ロールを自動的に割り当てます。

| エンティティ | デフォルトの Registry ロール |
| ----- | ----- |
| Teams | Viewer |
| ユーザー (管理者以外) | Viewer |
| 組織の管理者 | Admin |

Registry の管理者は、Registry 内の ユーザー および Teams のロールを割り当てまたは変更できます。
詳細については、[Registry での ユーザー ロールの設定]({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}}) を参照してください。

{{% alert title="W&B ロールの種類" %}}
W&B には、2 種類のロールがあります。[Teams ロール]({{< ref "/guides/models/app/settings-page/teams.md#team-role-and-permissions" >}}) と [Registry ロール]({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}}) です。

Teams でのロールは、Registry でのロールに影響を与えたり、関係したりすることはありません。
{{% /alert %}}

次の表に、ユーザー が持つことができるさまざまなロールと、その権限を示します。

| 権限                                                     | 権限グループ | Viewer | Member | Admin |
|--------------------------------------------------------------- |------------------|--------|--------|-------|
| コレクションの詳細を表示する                                    | Read             |   X    |   X    |   X   |
| リンクされた Artifact の詳細を表示する                               | Read             |   X    |   X    |   X   |
| 使用法: use_artifact を使用して Registry 内の Artifact を消費する     | Read             |   X    |   X    |   X   |
| リンクされた Artifact をダウンロードする                                     | Read             |   X    |   X    |   X   |
| Artifact のファイルビューアーからファイルをダウンロードする                  | Read             |   X    |   X    |   X   |
| Registry を検索する                                              | Read             |   X    |   X    |   X   |
| Registry の 設定 と ユーザー リストを表示する                       | Read             |   X    |   X    |   X   |
| コレクションの新しい自動化を作成する                       | Create           |        |   X    |   X   |
| 新しい バージョン が追加されたときに Slack 通知をオンにする        | Create           |        |   X    |   X   |
| 新しいコレクションを作成する                                        | Create           |        |   X    |   X   |
| 新しいカスタム Registry を作成する                                   | Create           |        |   X    |   X   |
| コレクションカード (説明) を編集する                             | Update           |        |   X    |   X   |
| リンクされた Artifact の説明を編集する                               | Update           |        |   X    |   X   |
| コレクションのタグを追加または削除する                               | Update           |        |   X    |   X   |
| リンクされた Artifact から エイリアス を追加または削除する                  | Update           |        |   X    |   X   |
| 新しい Artifact をリンクする                                            | Update           |        |   X    |   X   |
| Registry で許可されている種類のリストを編集する                         | Update           |        |   X    |   X   |
| カスタム Registry 名を編集する                                      | Update           |        |   X    |   X   |
| コレクションを削除する                                            | Delete           |        |   X    |   X   |
| 自動化を削除する                                           | Delete           |        |   X    |   X   |
| Registry から Artifact のリンクを解除する                             | Delete           |        |   X    |   X   |
| Registry で許可されている Artifact の種類を編集する                    | Admin            |        |        |   X   |
| Registry の表示設定 (Organization または Restricted) を変更する        | Admin            |        |        |   X   |
| Registry に ユーザー を追加する                                        | Admin            |        |        |   X   |
| Registry で ユーザー のロールを割り当てるか変更する                   | Admin            |        |        |   X   |

### 継承された権限

Registry での ユーザー の権限は、個人または Teams メンバーシップによって割り当てられた、その ユーザー に割り当てられた最高の特権レベルによって異なります。

たとえば、Registry の管理者が Nico という ユーザー を Registry A に追加し、**Viewer** Registry ロールを割り当てるとします。次に、Registry の管理者が Foundation Model Team という Teams を Registry A に追加し、Foundation Model Team に **Member** Registry ロールを割り当てます。

Nico は Foundation Model Team のメンバーであり、**Member** は Registry のメンバーです。**Member** は **Viewer** よりも多くの権限を持っているため、W&B は Nico に **Member** ロールを付与します。

次の表は、ユーザー の個々の Registry ロールと、メンバーである Teams の Registry ロールとの間で競合が発生した場合の、最高の権限レベルを示しています。

| Teams Registry ロール | 個々の Registry ロール | 継承された Registry ロール |
| ------ | ------ | ------ |
| Viewer | Viewer | Viewer |
| Member | Viewer | Member |
| Admin  | Viewer | Admin  |

競合がある場合、W&B は ユーザー の名前の横に最高の権限レベルを表示します。

たとえば、次の画像では、Alex は `smle-reg-team-1` Teams のメンバーであるため、**Member** ロールの特権を継承しています。

{{< img src="/images/registry/role_conflict.png" alt="A user inherits a Member role because they are part of a team." >}}

## Registry ロールの構成
1. Registry (https://wandb.ai/registry/) に移動します。
2. 構成する Registry を選択します。
3. 右上隅にある歯車アイコンをクリックします。
4. **Registry members and roles** セクションまでスクロールします。
5. **Member** フィールド内で、権限を編集する ユーザー または Teams を検索します。
6. **Registry role** 列で、ユーザー のロールをクリックします。
7. ドロップダウンから、ユーザー に割り当てるロールを選択します。