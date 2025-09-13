---
title: レジストリ アクセスの設定
menu:
  default:
    identifier: ja-guides-core-registry-configure_registry
    parent: registry
weight: 3
---

レジストリ管理者は、レジストリの [ロールを設定]({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}})、[ユーザーを追加]({{< relref path="configure_registry.md#add-a-user-or-a-team-to-a-registry" lang="ja" >}})、または[ユーザーを削除]({{< relref path="configure_registry.md#remove-a-user-or-team-from-a-registry" lang="ja" >}})できます。これは、レジストリの設定から行います。

## ユーザーを管理する

### ユーザーまたはチームを追加する

レジストリ管理者は、個々のユーザーまたはチーム全体をレジストリに追加できます。ユーザーまたはチームをレジストリに追加するには：

1. W&B App UI で **Registry** App に移動します。
2. ユーザーまたはチームを追加したいレジストリを選択します。
3. 右上隅の歯車アイコンをクリックして、レジストリの設定にアクセスします。
4. **Registry access** セクションで、**Add access** をクリックします。
5. **Include users and teams** フィールドに、1人以上のユーザー名、メールアドレス、またはチーム名を入力します。
6. **Add access** をクリックします。

{{< img src="/images/registry/add_team_registry.gif" alt="レジストリにチームを追加する" >}}

[レジストリでのユーザーロールの設定]({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}})、または [レジストリロールの権限]({{< relref path="configure_registry.md#registry-role-permissions" lang="ja" >}}) について詳しくはこちら。

### ユーザーまたはチームを削除する

レジストリ管理者は、個々のユーザーまたはチーム全体をレジストリから削除できます。ユーザーまたはチームをレジストリから削除するには：

1. W&B App UI で **Registry** App に移動します。
2. ユーザーを削除したいレジストリを選択します。
3. 右上隅の歯車アイコンをクリックして、レジストリの設定にアクセスします。
4. **Registry access** セクションに移動し、削除したいユーザー名、メールアドレス、またはチームを入力します。
5. **Delete** ボタンをクリックします。

{{% alert %}}
チームからユーザーを削除すると、そのユーザーのレジストリへのアクセスも削除されます。
{{% /alert %}}

### レジストリの所有者を変更する

レジストリ管理者は、**Restricted Viewer** や **Viewer** を含む任意のメンバーをレジストリの所有者に指定できます。レジストリの所有権は、主に説明責任を果たすことを目的としており、ユーザーに割り当てられたロールによって付与される権限を超えた追加の権限を与えるものではありません。

所有者を変更するには：
1. W&B App UI で **Registry** App に移動します。
2. 設定したいレジストリを選択します。
3. 右上隅の歯車アイコンをクリックします。
4. **Registry members and roles** セクションまでスクロールします。
5. メンバーの行にカーソルを合わせます。
6. 行の最後にある **...** アクションメニューをクリックし、**Make owner** をクリックします。

## レジストリロールを設定する

このセクションでは、Registry メンバーのロールを設定する方法について説明します。各ロールの機能、優先順位、デフォルトなど、Registry ロールの詳細については、[レジストリロールの詳細](#details-about-registry-roles) を参照してください。

1. W&B App UI で **Registry** App に移動します。
2. 設定したいレジストリを選択します。
3. 右上隅の歯車アイコンをクリックします。
4. **Registry members and roles** セクションまでスクロールします。
5. **Member** フィールド内で、権限を編集したいユーザーまたはチームを検索します。
6. **Registry role** 列で、ユーザーのロールをクリックします。
7. ドロップダウンから、ユーザーに割り当てたいロールを選択します。

## レジストリロールの詳細

以下のセクションでは、レジストリロールについてさらに詳しく説明します。

{{% alert %}}
チームでの [あなたのロール]({{< ref "/guides/models/app/settings-page/teams.md#team-role-and-permissions" >}}) は、いかなるレジストリでのあなたのロールにも影響せず、関連性もありません。
{{% /alert %}}

### デフォルトロール

W&B は、ユーザーまたはチームがレジストリに追加されたときに、デフォルトの **registry role** を自動的に割り当てます。このロールは、そのレジストリで何ができるかを決定します。

| Entities | デフォルトのレジストリロール<br />(専用クラウド / セルフマネージド) | デフォルトのレジストリロール<br />(Multi-Tenant Cloud) |
|---|---|---|
| Teams | Viewer | Restricted Viewer |
| Users またはサービスアカウント (管理者以外) | Viewer | Restricted Viewer |
| サービスアカウント (管理者以外) | Member<sup><a href="#service_account_footnote">1</a></sup> | Member<sup><a href="#service_account_footnote">1</a></sup> |
| 組織管理者 | Admin | Admin |

<a id="service_account_footnote">1</a>: サービスアカウントは **Viewer** または **Restricted Viewer** のロールを持つことはできません。

レジストリ管理者は、レジストリ内のユーザーとチームのロールを割り当てたり変更したりできます。
詳細については、[レジストリでのユーザーロールの設定]({{< relref path="configure_registry.md#configure-registry-roles" lang="ja" >}}) を参照してください。

{{% alert title="Restricted Viewer ロールの利用可能性" %}}
**Restricted Viewer** ロールは現在、Multi-Tenant Cloud 組織でのみ招待制で利用可能です。アクセスをリクエストしたり、専用クラウド または セルフマネージドでこの機能に関心を示す場合は、[サポートにお問い合わせください](mailto:support@wandb.ai)。

このロールは、コレクション、オートメーション、またはその他のレジストリリソースを作成、更新、または削除する機能なしに、レジストリの Artifacts への読み取り専用アクセスを提供します。

**Viewer** とは異なり、**Restricted Viewer** は：
- アーティファクトファイルをダウンロードしたり、ファイルコンテンツにアクセスしたりすることはできません。
- W&B SDK の `use_artifact()` を使用して Artifacts を利用することはできません。
{{% /alert %}}

### ロールの権限

以下の表は、各レジストリロールと、それぞれのロールによって提供される権限を示しています。

| 権限 | 権限グループ | Restricted Viewer<br />(Multi-Tenant Cloud、招待制) | Viewer | Member | Admin |
|---|---|---|---|---|---|
| コレクションの詳細を表示する | Read | ✓ | ✓ | ✓ | ✓ |
| リンクされたアーティファクトの詳細を表示する | Read | ✓ | ✓ | ✓ | ✓ |
| 利用: `use_artifact` を使用してレジストリ内のアーティファクトを利用する | Read | | ✓ | ✓ | ✓ |
| リンクされたアーティファクトをダウンロードする | Read | | ✓ | ✓ | ✓ |
| アーティファクトのファイルビューアからファイルをダウンロードする | Read | | ✓ | ✓ | ✓ |
| レジストリを検索する | Read | ✓ | ✓ | ✓ | ✓ |
| レジストリの設定とユーザーリストを表示する | Read | ✓ | ✓ | ✓ | ✓ |
| コレクションの新しいオートメーションを作成する | Create | | | ✓ | ✓ |
| 新しいバージョンが追加された際の Slack 通知をオンにする | Create | | | ✓ | ✓ |
| 新しいコレクションを作成する | Create | | | ✓ | ✓ |
| 新しいカスタムレジストリを作成する | Create | | | ✓ | ✓ |
| コレクションカードを編集する (説明) | Update | | | ✓ | ✓ |
| リンクされたアーティファクトの説明を編集する | Update | | | ✓ | ✓ |
| コレクションのタグを追加または削除する | Update | | | ✓ | ✓ |
| リンクされたアーティファクトからエイリアスを追加または削除する | Update | | | ✓ | ✓ |
| 新しいアーティファクトをリンクする | Update | | | ✓ | ✓ |
| レジストリの許可されたタイプリストを編集する | Update | | | ✓ | ✓ |
| カスタムレジストリ名を編集する | Update | | | ✓ | ✓ |
| コレクションを削除する | Delete | | | ✓ | ✓ |
| オートメーションを削除する | Delete | | | ✓ | ✓ |
| レジストリからアーティファクトのリンクを解除する | Delete | | | ✓ | ✓ |
| レジストリで承認されたアーティファクトのタイプを編集する | Admin | | | | ✓ |
| レジストリの公開範囲を変更する (組織または制限付き) | Admin | | | | ✓ |
| レジストリにユーザーを追加する | Admin | | | | ✓ |
| レジストリでのユーザーのロールを割り当てるか変更する | Admin | | | | ✓ |

### 継承されたレジストリロール

レジストリのメンバーシップリストには、各ユーザーの継承された (実効) レジストリロール (薄い灰色で) が、その行のロールのドロップダウンの横に表示されます。

{{< img src="/images/registry/role_conflict.png" alt="ユーザーの実効レジストリロールを示すレジストリメンバーシップリスト" >}}

特定のレジストリにおけるユーザーの実効ロールは、組織、レジストリ、およびレジストリを所有するチームにおけるそのユーザーのロールの中で、継承されたものか明示的に割り当てられたものかを問わず、 _最も高い_ ロールと一致します。例：

- チームが所有する特定のレジストリで **Viewer** ロールを持つチームの **Admin** または組織の **Admin** は、実質的にそのレジストリの **Admin** です。
- チームで **Member** ロールを持つレジストリの **Viewer** は、実質的にそのレジストリの **Member** です。
- 特定のレジストリで **Member** ロールを持つチームの **Viewer** は、実質的にそのレジストリの **Member** です。

### SDK の互換性

{{% alert title="SDK バージョンの要件" %}}
**Restricted Viewer** として Artifacts にアクセスするために W&B SDK を使用するには、W&B SDK バージョン 0.19.9 以降を使用する必要があります。そうしないと、一部の SDK コマンドで権限エラーが発生します。
{{% /alert %}}

**Restricted Viewer** が SDK を使用する場合、特定の機能は利用できないか、異なる動作をします。

以下のメソッドは利用できず、権限エラーが発生します。
- [`Run.use_artifact()`]({{< relref path="/ref/python/sdk/classes/run/#method-runuse_artifact" lang="ja" >}})
- [`Artifact.download()`]({{< relref path="/ref/python/sdk/classes/artifact/#method-artifactdownload" lang="ja" >}})
- [`Artifact.file()`]({{< relref path="/ref/python/sdk/classes/artifact/#method-artifactfile" lang="ja" >}})
- [`Artifact.files()`]({{< relref path="/ref/python/sdk/classes/artifact/#method-artifactfiles" lang="ja" >}})

以下のメソッドはアーティファクトのメタデータに限定されます。
- [`Artifact.get_entry()`]({{< relref path="/ref/python/sdk/classes/artifact/#method-artifactget_entry" lang="ja" >}})
- [`Artifact.get_path()`]({{< relref path="/ref/python/sdk/classes/artifact/#method-artifactget_path" lang="ja" >}})
- [`Artifact.get()`]({{< relref path="/ref/python/sdk/classes/artifact/#method-artifactget" lang="ja" >}})
- [`Artifact.verify()`]({{< relref path="/ref/python/sdk/classes/artifact/#method-artifactverify" lang="ja" >}})

### クロスレジストリ権限

ユーザーは異なるレジストリで異なるロールを持つことができます。例えば、ユーザーはレジストリ A では **Restricted Viewer** ですが、レジストリ B では **Viewer** であることができます。この場合：

- 両方のレジストリにリンクされた同じアーティファクトは、異なるアクセスレベルを持つことになります。
- レジストリ A では、ユーザーは **Restricted Viewer** であり、ファイルをダウンロードしたりアーティファクトを使用したりすることはできません。
- レジストリ B では、ユーザーは **Viewer** であり、ファイルをダウンロードしたりアーティファクトを使用したりすることができます。
- 言い換えれば、アクセスはアーティファクトがアクセスされるレジストリによって決定されます。