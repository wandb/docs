---
title: レジストリへのアクセスを設定
menu:
  default:
    identifier: configure_registry
    parent: registry
weight: 3
---

レジストリの管理者は、レジストリの設定を変更することで、[レジストリのロールを設定]({{< relref "configure_registry.md#configure-registry-roles" >}})したり、[ユーザーを追加]({{< relref "configure_registry.md#add-a-user-or-a-team-to-a-registry" >}})・[ユーザーを削除]({{< relref "configure_registry.md#remove-a-user-or-team-from-a-registry" >}})したりできます。

## ユーザーの管理

### ユーザーまたはチームの追加

レジストリ管理者は、個別のユーザーやチーム全体をレジストリに追加できます。ユーザーやチームをレジストリに追加する手順は以下の通りです。

1. https://wandb.ai/registry/ にアクセスします。
2. ユーザーやチームを追加したいレジストリを選択します。
3. 画面右上の歯車アイコンをクリックし、レジストリの設定にアクセスします。
4. **Registry access（レジストリアクセス）** セクションで **Add access（アクセスの追加）** をクリックします。
5. **Include users and teams（ユーザー・チームを含める）** フィールドに、追加したいユーザー名・メールアドレス・チーム名を入力します。
6. **Add access（アクセスの追加）** をクリックします。

{{< img src="/images/registry/add_team_registry.gif" alt="Adding teams to registry" >}}

[レジストリ内のユーザーロールの設定方法]({{< relref "configure_registry.md#configure-registry-roles" >}})や、[Registryロールの権限]({{< relref "configure_registry.md#registry-role-permissions" >}})についてもご覧ください。

### ユーザーやチームの削除

レジストリ管理者は、個人のユーザーやチーム全体をレジストリから削除することができます。削除手順は以下の通りです。

1. https://wandb.ai/registry/ にアクセスします。
2. ユーザーを削除したいレジストリを選択します。
3. 画面右上の歯車アイコンをクリックしてレジストリ設定に入ります。
4. **Registry access** セクションで、削除したいユーザー名・メールアドレス・チーム名を入力します。
5. **Delete** ボタンをクリックします。

{{% alert %}}
チームからユーザーを削除すると、そのユーザーのレジストリアクセスも削除されます。
{{% /alert %}}

## Registry ロール

レジストリ内の各ユーザーには *registry ロール* が割り当てられ、そのレジストリでどのような操作ができるかが決まります。

W&B は、ユーザーやチームがレジストリに追加された際に、自動的にデフォルトの registry ロールを割り当てます。

| Entity | デフォルト registry ロール |
| ------ | -------------------- |
| Team | Viewer |
| User（管理者以外） | Viewer |
| Org admin | Admin |

レジストリの管理者は、レジストリ内のユーザーやチームに対してロールの割り当てや変更ができます。 詳しくは [レジストリ内のユーザーロールの設定方法]({{< relref "configure_registry.md#configure-registry-roles" >}}) をご覧ください。

{{% alert title="W&Bのロール種別" %}}
W&B には [Team ロール]({{< ref "/guides/models/app/settings-page/teams.md#team-role-and-permissions" >}}) と [Registry ロール]({{< relref "configure_registry.md#configure-registry-roles" >}}) の2種類があります。

チームでのロールと、各レジストリにおけるロールは相互に影響しません。
{{% /alert %}}

以下の表は、各ロールが持つ権限の違いをまとめたものです。

| 権限                                                     | パーミッショングループ | Viewer | Member | Admin |
|---------------------------------------------------------|-----------------------|--------|--------|-------|
| コレクションの詳細を閲覧                                 | Read                  |   X    |   X    |   X   |
| リンク済み Artifacts の詳細を閲覧                        | Read                  |   X    |   X    |   X   |
| use_artifact でレジストリ内の Artifact を利用            | Read                  |   X    |   X    |   X   |
| リンク済み Artifact のダウンロード                       | Read                  |   X    |   X    |   X   |
| Artifact ファイルビューアからのファイルダウンロード       | Read                  |   X    |   X    |   X   |
| レジストリの検索                                         | Read                  |   X    |   X    |   X   |
| レジストリの設定やユーザーリストの閲覧                   | Read                  |   X    |   X    |   X   |
| コレクション用の新しい自動化設定を作成                   | Create                |        |   X    |   X   |
| 新バージョン追加時の Slack 通知のON/OFF                  | Create                |        |   X    |   X   |
| 新規コレクションの作成                                   | Create                |        |   X    |   X   |
| カスタムレジストリの新規作成                             | Create                |        |   X    |   X   |
| コレクションカード（説明）の編集                         | Update                |        |   X    |   X   |
| リンク済み Artifact の説明編集                           | Update                |        |   X    |   X   |
| コレクションのタグ追加・削除                             | Update                |        |   X    |   X   |
| リンク済み Artifact のエイリアス追加・削除               | Update                |        |   X    |   X   |
| 新しい Artifact のリンク                                 | Update                |        |   X    |   X   |
| レジストリの許可タイプリスト編集                         | Update                |        |   X    |   X   |
| カスタムレジストリ名の編集                               | Update                |        |   X    |   X   |
| コレクションの削除                                       | Delete                |        |   X    |   X   |
| 自動化設定の削除                                         | Delete                |        |   X    |   X   |
| レジストリから Artifact のリンク解除                     | Delete                |        |   X    |   X   |
| レジストリで許可する Artifact タイプの編集               | Admin                 |        |        |   X   |
| レジストリの公開範囲変更（Organization または Restricted）| Admin                 |        |        |   X   |
| レジストリへのユーザー追加                                | Admin                 |        |        |   X   |
| レジストリ内のユーザーのロール割り当て・変更              | Admin                 |        |        |   X   |

### 権限の継承

ユーザーのレジストリ内の権限は、そのユーザー個人またはチームメンバーとして与えられた権限のうち、より高い方が適用されます。

例えば、Nico さんというユーザーを **Viewer** として Registry A に追加した後、Foundation Model Team というチームを Registry A に **Member** として追加したとします。

Nico さんが Foundation Model Team のメンバーなら、そのチームが **Member** 権限を持つため、Nico さんにも **Member** 権限が付与されます（**Member** の方が **Viewer** より強い権限です）。

以下の表は、個人とチームで異なるロールが割り当てられている場合の、「より高い権限に引き上げられる」仕組みを示したものです。

| チームの Registry ロール | 個人の Registry ロール | 継承される Registry ロール |
| --------- | ---------------- | ------------------- |
| Viewer    | Viewer           | Viewer              |
| Member    | Viewer           | Member              |
| Admin     | Viewer           | Admin               |

このように競合があった場合、W&B の画面上では「実際に適用される一番強い権限」がユーザー名の横に表示されます。

例えば以下の画像のように、Alex さんは `smle-reg-team-1` チームのメンバーなので **Member** ロールの権限を継承しています。

{{< img src="/images/registry/role_conflict.png" alt="Registry role conflict resolution" >}}

## レジストリロールの設定
1. https://wandb.ai/registry/ にアクセスします。
2. 設定を変更したいレジストリを選択します。
3. 画面右上の歯車アイコンをクリックします。
4. **Registry members and roles** セクションまでスクロールします。
5. **Member** の欄で、権限を編集したいユーザーやチームを検索します。
6. **Registry role** カラムで、該当ユーザーの現在のロールをクリックします。
7. ドロップダウンから、割り当てたいロールを選択します。