---
title: Configure registry access
menu:
  default:
    identifier: ja-guides-models-registry-configure_registry
    parent: registry
weight: 3
---

Registry の管理者は、Registry の [設定] ({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}}) を構成することで、Registry から [Registry ロールを設定] ({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}}) したり、[ユーザーを追加] ({{< relref path="configure_registry.md#add-a-user-or-a-team-to-a-registry" lang="ja" >}}) したり、[ユーザーを削除] ({{< relref path="configure_registry.md#remove-a-user-or-team-from-a-registry" lang="ja" >}}) したりできます。

## ユーザーの管理

### ユーザーまたは Team の追加

Registry の管理者は、個々の ユーザー または Team 全体を Registry に追加できます。ユーザー または Team を Registry に追加するには、次の手順に従います。

1. Registry (https://wandb.ai/registry/) に移動します。
2. ユーザー または Team を追加する Registry を選択します。
3. 右上隅にある歯車アイコンをクリックして、Registry の 設定 にアクセスします。
4. **Registry access** セクションで、**Add access** をクリックします。
5. **Include users and teams** フィールドに、1 つ以上の ユーザー 名、メールアドレス、または Team 名を指定します。
6. **Add access** をクリックします。

{{< img src="/images/registry/add_team_registry.gif" alt="UI を使用して Team および個々の ユーザー を Registry に追加するアニメーション" >}}

[Registry での ユーザー ロールの設定] ({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}}) 、または [Registry ロールの権限] ({{< relref path="configure_registry.md#registry-role-permissions" lang="ja" >}}) について詳しくはこちらをご覧ください。

### ユーザー または Team の削除
Registry の管理者は、個々の ユーザー または Team 全体を Registry から削除できます。ユーザー または Team を Registry から削除するには、次の手順に従います。

1. Registry (https://wandb.ai/registry/) に移動します。
2. ユーザー を削除する Registry を選択します。
3. 右上隅にある歯車アイコンをクリックして、Registry の 設定 にアクセスします。
4. **Registry access** セクションに移動し、削除する ユーザー 名、メールアドレス、または Team を入力します。
5. **Delete** ボタンをクリックします。

{{% alert %}}
Team から ユーザー を削除すると、その ユーザー の Registry へのアクセス権も削除されます。
{{% /alert %}}

## Registry ロール

Registry 内の各 ユーザー は、*Registry ロール* を持っており、そのロールによって、その Registry で何ができるかが決まります。

W&B は、ユーザー または Team が Registry に追加されると、デフォルトの Registry ロールを自動的に割り当てます。

| エンティティ | デフォルトの Registry ロール |
| ----- | ----- |
| Team | Viewer |
| ユーザー (管理者以外) | Viewer |
| Org admin | Admin |

Registry の管理者は、Registry 内の ユーザー および Team のロールを割り当てたり、変更したりできます。
詳細については、[Registry での ユーザー ロールの設定] ({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}}) を参照してください。

{{% alert title="W&B ロールの種類" %}}
W&B には、2 種類のロールがあります。[Team ロール] ({{< ref "/guides/models/app/settings-page/teams.md#team-role-and-permissions" >}}) と [Registry ロール] ({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}}) です。

Team でのロールは、Registry でのロールに影響を与えたり、関係したりすることはありません。
{{% /alert %}}

次の表に、ユーザー が持つことができるさまざまなロールとその権限を示します。

| 権限                                                         | 権限グループ | Viewer | Member | Admin |
|--------------------------------------------------------------- |------------------|--------|--------|-------|
| Collection の詳細を表示する                                       | Read             |   X    |   X    |   X   |
| リンクされた Artifact の詳細を表示する                                   | Read             |   X    |   X    |   X   |
| 使用法: use_artifact を使用して Registry 内の Artifact を消費する        | Read             |   X    |   X    |   X   |
| リンクされた Artifact をダウンロードする                                    | Read             |   X    |   X    |   X   |
| Artifact のファイルビューアーからファイルをダウンロードする                    | Read             |   X    |   X    |   X   |
| Registry を検索する                                                | Read             |   X    |   X    |   X   |
| Registry の 設定 と ユーザー リストを表示する                                   | Read             |   X    |   X    |   X   |
| Collection の新しい自動化を作成する                                     | Create           |        |   X    |   X   |
| 新しい バージョン の追加に関する Slack 通知をオンにする                          | Create           |        |   X    |   X   |
| 新しい Collection を作成する                                         | Create           |        |   X    |   X   |
| 新しいカスタム Registry を作成する                                    | Create           |        |   X    |   X   |
| Collection カード (説明) を編集する                                   | Update           |        |   X    |   X   |
| リンクされた Artifact の説明を編集する                                  | Update           |        |   X    |   X   |
| Collection の タグ を追加または削除する                                   | Update           |        |   X    |   X   |
| リンクされた Artifact から エイリアス を追加または削除する                          | Update           |        |   X    |   X   |
| 新しい Artifact をリンクする                                           | Update           |        |   X    |   X   |
| Registry の許可された型のリストを編集する                                    | Update           |        |   X    |   X   |
| カスタム Registry 名を編集する                                      | Update           |        |   X    |   X   |
| Collection を削除する                                               | Delete           |        |   X    |   X   |
| 自動化を削除する                                                   | Delete           |        |   X    |   X   |
| Registry から Artifact のリンクを解除する                                  | Delete           |        |   X    |   X   |
| Registry に受け入れられる Artifact の種類を編集する                               | Admin            |        |        |   X   |
| Registry の可視性 (組織または制限付き) を変更する                       | Admin            |        |        |   X   |
| Registry に ユーザー を追加する                                          | Admin            |        |        |   X   |
| Registry 内の ユーザー のロールを割り当てるか、変更する                               | Admin            |        |        |   X   |

### 継承された権限

Registry 内の ユーザー の権限は、個人または Team メンバーシップによって割り当てられた、その ユーザー に割り当てられた最高の権限レベルによって異なります。

たとえば、Registry の管理者が Nico という ユーザー を Registry A に追加し、**Viewer** Registry ロールを割り当てたとします。次に、Registry の管理者が Foundation Model Team という Team を Registry A に追加し、Foundation Model Team に **Member** Registry ロールを割り当てます。

Nico は Foundation Model Team のメンバーであり、これは Registry の **Member** です。**Member** は **Viewer** よりも多くの権限を持っているため、W&B は Nico に **Member** ロールを付与します。

次の表は、ユーザー の個々の Registry ロールと、その ユーザー がメンバーである Team の Registry ロールとの間に競合が発生した場合の、最高の権限レベルを示しています。

| Team Registry ロール | 個々の Registry ロール | 継承された Registry ロール |
| ------ | ------ | ------ |
| Viewer | Viewer | Viewer |
| Member | Viewer | Member |
| Admin  | Viewer | Admin  |

競合がある場合、W&B は ユーザー 名の横に最高レベルの権限を表示します。

たとえば、次の図では、Alex は `smle-reg-team-1` Team のメンバーであるため、**Member** ロールの権限を継承しています。

{{< img src="/images/registry/role_conflict.png" alt="ユーザー は Team の一部であるため、Member ロールを継承します。" >}}

## Registry ロールの設定
1. Registry (https://wandb.ai/registry/) に移動します。
2. 設定する Registry を選択します。
3. 右上隅にある歯車アイコンをクリックします。
4. **Registry members and roles** セクションまでスクロールします。
5. **Member** フィールド内で、権限を編集する ユーザー または Team を検索します。
6. **Registry role** 列で、ユーザー のロールをクリックします。
7. ドロップダウンから、ユーザー に割り当てるロールを選択します。
