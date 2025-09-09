---
title: 組織の管理
menu:
  default:
    identifier: ja-guides-hosting-iam-access-management-manage-organization
    parent: access-management
weight: 1
---

組織の管理者は、組織内の [Users を管理]({{< relref path="#add-and-manage-users" lang="ja" >}}) したり、[Teams を管理]({{< relref path="#add-and-manage-teams" lang="ja" >}}) できます。

チーム管理者は [Teams を管理]({{< relref path="#add-and-manage-teams" lang="ja" >}}) できます。

{{% alert %}}
以下のワークフローは、インスタンス管理者ロールを持つ Users に適用されます。自分にインスタンス管理者権限が必要だと思われる場合は、組織内の管理者に連絡してください。
{{% /alert %}}

組織でのユーザー管理を簡素化したい場合は、[ユーザーとチームの管理を自動化]({{< relref path="../automate_iam.md" lang="ja" >}}) を参照してください。




## 組織名を変更する
{{% alert %}}
以下のワークフローは W&B のマルチテナント SaaS クラウドにのみ適用されます。
{{% /alert %}}

1. https://wandb.ai/home に移動します。
2. ページ右上の **User menu** ドロップダウンを選択します。ドロップダウンの **Account** セクションで **Settings** を選択します。
3. **Settings** タブの中で **General** を選択します。
4. **Change name** ボタンを選択します。
5. 表示されるモーダルで組織の新しい名前を入力し、**Save name** ボタンを選択します。

## Users を追加・管理する

管理者は組織のダッシュボードを使って以下を行えます:
- Users を招待または削除する。
- user の組織ロールを割り当てまたは更新し、カスタムロールを作成する。
- 課金管理者を割り当てる。

組織管理者が Users を組織に追加する方法はいくつかあります:

1. 招待による参加
2. SSO による自動プロビジョニング
3. ドメインキャプチャ

### シートと料金

以下の表は Models と Weave におけるシートの仕組みをまとめたものです:

| Product | Seats | 課金の基準 |
| ----- | ----- | ----- |
| Models | シート単位の課金 | 有料の Models シート数と蓄積した使用量に基づいて、サブスクリプション費用が決まります。各 user には Full、Viewer、No access の 3 種類のいずれかのシートタイプを割り当てられます。 |
| Weave | 無料 | 使用量ベース |

### user を招待する

管理者は、組織や組織内の特定の Teams に Users を招待できます。

{{< tabpane text=true >}}
{{% tab header="マルチテナント SaaS クラウド" value="saas" %}}
1. https://wandb.ai/home に移動します。
1. ページ右上の **User menu** ドロップダウンを選択します。ドロップダウンの **Account** セクションで **Users** を選択します。
3. **Invite new user** を選択します。
4. 表示されるモーダルの **Email or username** フィールドに、その user のメールアドレスまたはユーザー名を入力します。
5. （推奨）**Choose teams** ドロップダウンから、user を参加させる team を追加します。
6. **Select role** ドロップダウンから、その user に割り当てるロールを選択します。user のロールは後で変更できます。可能なロールについては、[ロールを割り当てる]({{< relref path="#assign-or-update-a-team-members-role" lang="ja" >}}) にある表を参照してください。
7. **Send invite** ボタンを選択します。

**Send invite** を選択すると、W&B はサードパーティのメールサーバーを使用して招待リンクをその user のメールアドレス宛てに送信します。招待を受け入れると、user はあなたの組織にアクセスできます。
{{% /tab %}}

{{% tab header="専用クラウドまたは Self-managed" value="dedicated"%}}
1. `https://<org-name>.io/console/settings/` に移動します。`<org-name>` はあなたの組織名に置き換えてください。
2. **Add user** ボタンを選択します。
3. 表示されるモーダルの **Email** フィールドに、新しい user のメールアドレスを入力します。
4. **Role** ドロップダウンから、その user に割り当てるロールを選択します。user のロールは後で変更できます。可能なロールについては、[ロールを割り当てる]({{< relref path="#assign-or-update-a-team-members-role" lang="ja" >}}) にある表を参照してください。
5. W&B からサードパーティのメールサーバー経由で招待リンクを送信したい場合は、**Send invite email to user** にチェックを入れます。
6. **Add new user** ボタンを選択します。
{{% /tab %}}
{{< /tabpane >}}

### Users を自動プロビジョニングする

SSO を設定し、SSO プロバイダが許可している場合、メールドメインが一致する W&B user は SSO を使ってあなたの W&B Organization にサインインできます。SSO はすべてのエンタープライズライセンスで利用可能です。

{{% alert title="認証に SSO を有効化する" %}}
W&B は、Single Sign-On (SSO) による認証を強く推奨します。あなたの組織で SSO を有効化するには、W&B チームに連絡してください。

専用クラウドまたは Self-managed インスタンスでの SSO 設定方法については、[SSO with OIDC]({{< relref path="../authentication/sso.md" lang="ja" >}}) または [SSO with LDAP]({{< relref path="../authentication/ldap.md" lang="ja" >}}) を参照してください。{{% /alert %}} 

W&B は、自動プロビジョニングされた Users にデフォルトで "Member" ロールを付与します。自動プロビジョニングされた Users のロールはいつでも変更できます。

SSO による自動プロビジョニングは、専用クラウドインスタンスおよび Self-managed デプロイメントではデフォルトで有効です。自動プロビジョニングはオフにできます。オフにすると、特定の Users のみを選択的に W&B の組織へ追加できます。

以下のタブでは、デプロイメントタイプ別に自動プロビジョニングをオフにする方法を説明します:

{{< tabpane text=true >}}
{{% tab header="専用クラウド" value="dedicated" %}}
専用クラウドインスタンスで自動プロビジョニングをオフにしたい場合は、W&B チームに連絡してください。
{{% /tab %}}

{{% tab header="Self-managed" value="self_managed" %}}
W&B Console で SSO による自動プロビジョニングをオフにします:

1. `https://<org-name>.io/console/settings/` に移動します。`<org-name>` はあなたの組織名に置き換えてください。
2. **Security** を選択します。
3. **Disable SSO Provisioning** を選択して、SSO による自動プロビジョニングをオフにします。




{{% /tab %}}
{{< /tabpane >}}

{{% alert title="" %}}
SSO による自動プロビジョニングは、組織に大規模に Users を追加するのに有用です。組織管理者は個別の招待を作成する必要がありません。
{{% /alert %}}

### カスタムロールを作成する
{{% alert %}}
専用クラウドまたは Self-managed デプロイメントでカスタムロールを作成・割り当てるには、エンタープライズライセンスが必要です。
{{% /alert %}}

組織管理者は、View-Only もしくは Member ロールをベースに新しいロールを作成し、権限を追加してきめ細かなアクセス制御を実現できます。チーム管理者は、カスタムロールをチームメンバーに割り当てることができます。カスタムロールは組織レベルで作成されますが、割り当てはチームレベルで行います。

カスタムロールを作成する手順:

{{< tabpane text=true >}}
{{% tab header="マルチテナント SaaS クラウド" value="saas" %}}
1. https://wandb.ai/home に移動します。
1. ページ右上の **User menu** ドロップダウンを選択します。ドロップダウンの **Account** セクションで **Settings** を選択します。
1. **Roles** をクリックします。
1. **Custom roles** セクションで **Create a role** をクリックします。
1. ロール名を入力します。必要に応じて説明も入力します。
1. ベースとなるロールとして **Viewer** または **Member** を選択します。
1. 権限を追加するには、**Search permissions** フィールドをクリックし、追加したい権限を 1 つ以上選択します。
1. そのロールが持つ権限の要約を **Custom role permissions** セクションで確認します。
1. **Create Role** をクリックします。
{{% /tab %}}

{{% tab header="専用クラウドまたは Self-managed" value="dedicated"%}}
W&B Console で SSO による自動プロビジョニングをオフにします:

1. `https://<org-name>.io/console/settings/` に移動します。`<org-name>` はあなたの組織名に置き換えてください。
1. **Custom roles** セクションで **Create a role** をクリックします。
1. ロール名を入力します。必要に応じて説明も入力します。
1. ベースとなるロールとして **Viewer** または **Member** を選択します。
1. 権限を追加するには、**Search permissions** フィールドをクリックし、追加したい権限を 1 つ以上選択します。
1. そのロールが持つ権限の要約を **Custom role permissions** セクションで確認します。
1. **Create Role** をクリックします。

{{% /tab %}}
{{< /tabpane >}}

これで、チーム管理者は [Team settings]({{< relref path="#invite-users-to-a-team" lang="ja" >}}) からチームメンバーにカスタムロールを割り当てられます。

### ドメインキャプチャ
ドメインキャプチャは、従業員が自社の組織に参加できるようにし、新しい Users が会社の管理外でアセットを作成しないようにするための機能です。

{{% alert title="ドメインは一意である必要があります" %}}
ドメインは一意の識別子です。つまり、すでに別の組織で使用されているドメインは使用できません。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="マルチテナント SaaS クラウド" value="saas" %}}
ドメインキャプチャを使うと、`@example.com` のような会社のメールアドレスを持つ人を、W&B の SaaS クラウド組織に自動的に追加できます。これにより、従業員が正しい組織に参加でき、新しい Users が会社の管理外でアセットを作成してしまうことを防げます。

以下の表は、ドメインキャプチャの有無での新規および既存の Users の振る舞いをまとめたものです:

| | ドメインキャプチャあり | ドメインキャプチャなし |
| ----- | ----- | ----- |
| 新規 Users | 検証済みドメインから W&B にサインアップした Users は、自動的に組織のデフォルト team にメンバーとして追加されます。team への参加機能を有効にしている場合は、サインアップ時に追加の team を選べます。招待があれば、他の組織や team にも参加できます。 | 中央集約された組織があると知らずに、W&B アカウントを作成できてしまいます。 |
| 招待された Users | 招待を受け入れると自動的に組織に参加します。組織のデフォルト team に自動追加はされません。招待があれば、他の組織や team にも参加できます。 | 招待を受け入れると自動的に組織に参加します。招待があれば、他の組織や team にも参加できます。 |
| 既存 Users | あなたのドメイン由来の検証済みメールアドレスを持つ既存 Users は、W&B アプリ内で組織の Teams に参加できます。参加前に既存 Users が作成したデータはそのまま残ります。W&B は既存 user のデータを移行しません。 | 既存の W&B Users は複数の組織や team に散らばっている可能性があります。 |

招待されていない新規 Users を、参加時にデフォルトの team に自動割り当てするには:

1. https://wandb.ai/home に移動します。
2. 右上の **User menu** ドロップダウンを選択し、**Settings** を選びます。
3. **Settings** タブの中で **General** を選択します。
4. **Domain capture** の **Claim domain** ボタンを選択します。
5. **Default team** ドロップダウンから、新規 Users を自動参加させたい team を選択します。team が表示されない場合は、team の設定を更新する必要があります。[Teams を追加・管理]({{< relref path="#add-and-manage-teams" lang="ja" >}}) の手順を参照してください。
6. **Claim email domain** ボタンをクリックします。

招待されていない新規 Users を特定の team に自動割り当てするには、事前にその team の設定でドメインマッチングを有効化する必要があります。

1. `https://wandb.ai/<team-name>` にある team のダッシュボードへ移動します。`<team-name>` は有効化したい team 名です。
2. team のダッシュボード左側のグローバルナビゲーションで **Team settings** を選択します。
3. **Privacy** セクション内で「サインアップ時にメールドメインが一致する新規 Users にこの team への参加を推奨する」オプションを有効にします。

{{% /tab %}}
{{% tab header="専用クラウドまたは Self-managed" value="dedicated" %}}
専用クラウドまたは Self-managed デプロイメントをお使いの場合は、ドメインキャプチャの設定について W&B のアカウントチームに連絡してください。設定後、SaaS インスタンス上で会社のメールアドレスを使って W&B アカウントを作成した Users に対し、専用クラウドまたは Self-managed インスタンスへのアクセスを管理者へ依頼するよう自動的に促します。

| | ドメインキャプチャあり | ドメインキャプチャなし |
| ----- | ----- | -----|
| 新規 Users | SaaS クラウド上で検証済みドメインからサインアップした Users には、カスタマイズしたメールアドレス宛に管理者へ連絡するよう自動で案内します。製品を試用するために SaaS クラウド上で組織を作成することは引き続き可能です。 | 会社に専用インスタンスがあることを知らずに、W&B の SaaS クラウドアカウントを作成できてしまいます。 |
| 既存 Users | 既存の W&B Users は複数の組織や team に散在している可能性があります。| 既存の W&B Users は複数の組織や team に散在している可能性があります。|
{{% /tab %}}
{{< /tabpane >}}


### user のロールを割り当て・更新する

Organization 内のすべてのメンバーには、W&B Models と Weave の両方について、組織ロールとシートが割り当てられます。保有するシートタイプにより、課金ステータスと各プロダクトで実行できる操作が決まります。

組織に user を招待するときに、その user の組織ロールを最初に割り当てます。user のロールは後からいつでも変更できます。

組織内の user には、以下のいずれかのロールを割り当てられます:

| Role | 説明 |
| ----- | ----- |
| admin | 組織に他の Users を追加・削除し、user ロールを変更し、カスタムロールを管理し、Teams を追加するなどの操作ができるインスタンス管理者。管理者が不在の場合に備えて、admin を複数名にしておくことを W&B は推奨します。 |
| Member | インスタンス管理者に招待された、組織の一般 user。組織内の他の Users を招待したり管理したりすることはできません。 |
| Viewer (Enterprise のみ) | インスタンス管理者に招待された、組織の閲覧専用 user。所属する Teams と組織に対して読み取り専用のアクセスのみを持ちます。 |
| Custom Roles (Enterprise のみ) | カスタムロールにより、組織管理者は View-Only または Member を継承して新しいロールを作成し、権限を追加してきめ細かなアクセス制御を実現できます。チーム管理者は、それらのカスタムロールを自身のチームの Users に割り当てられます。|

user のロールを変更するには:

1. https://wandb.ai/home に移動します。
2. ページ右上の **User menu** ドロップダウンを選択し、**Users** を選びます。
4. 検索バーにその user の名前またはメールアドレスを入力します。
4. user 名の横にある **TEAM ROLE** ドロップダウンからロールを選択します。

### user のアクセスを割り当て・更新する

組織内の user には、Models のシートまたは Weave のアクセス種別として、Full、Viewer、No access のいずれかが割り当てられます。

| Seat type | 説明 |
| ----- | ----- |
| Full | このタイプの user は、Models または Weave に対して読み書きとエクスポートの完全な権限を持ちます。 |
| Viewer | 組織の閲覧専用 user。所属する組織および Teams に対して読み取り専用のアクセス、Models または Weave に対して閲覧専用のアクセスのみを持ちます。 |
| No access | このロールの user は、Models または Weave へのアクセス権を持ちません。 |

Models のシートタイプと Weave のアクセス種別は組織レベルで定義され、team に継承されます。user のシートタイプを変更したい場合は、組織の設定に移動して以下の手順に従ってください:

1. SaaS の場合、`https://wandb.ai/account-settings/<organization>/settings` にある組織の設定に移動します。山括弧（`<>`）内はあなたの組織名に置き換えてください。専用クラウドや Self-managed の場合は、`https://<your-instance>.wandb.io/org/dashboard` に移動します。
2. **Users** タブを選択します。
3. **Role** ドロップダウンから、その user に割り当てるシートタイプを選択します。

{{% alert %}}
組織ロールとサブスクリプション種別により、組織で利用可能なシートタイプが決まります。
{{% /alert %}}

### user を削除する

1. https://wandb.ai/home に移動します。
2. ページ右上の **User menu** ドロップダウンを選択し、**Users** を選びます。
4. 検索バーにその user の名前またはメールアドレスを入力します。
5. 名前の横に表示される三点リーダーアイコン (**...**) を選択します。
6. ドロップダウンから **Remove member** を選びます。

### 課金管理者を割り当てる
1. https://wandb.ai/home に移動します。
2. ページ右上の **User menu** ドロップダウンを選択し、**Users** を選びます。
4. 検索バーにその user の名前またはメールアドレスを入力します。
5. **Billing admin** 列で、課金管理者に割り当てたい user を選択します。


## Teams を追加・管理する
組織のダッシュボードを使って、組織内に Teams を作成・管理します。組織管理者またはチーム管理者は以下を行えます:
- Users を team に招待する、または team から Users を削除する。
- チームメンバーのロールを管理する。
- 新規 Users が組織に参加したときに team へ自動的に追加する。
- `https://wandb.ai/<team-name>` にある team のダッシュボードでチームのストレージを管理する。

### team を作成する

組織のダッシュボードから team を作成します:

1. https://wandb.ai/home に移動します。
2. 左側のナビゲーションパネルの **Teams** の下にある **Create a team to collaborate** を選択します。
{{< img src="/images/hosting/create_new_team.png" alt="新しい team を作成" >}}
3. 表示されるモーダルの **Team name** フィールドに team 名を入力します。
4. ストレージタイプを選択します。
5. **Create team** ボタンを選択します。

**Create team** を選択すると、W&B は `https://wandb.ai/<team-name>` の新しい team ページにリダイレクトします。`<team-name>` は team 作成時に指定した名前です。

team を作成したら、その team に Users を追加できます。

### Users を team に招待する

組織内の team に Users を招待します。team のダッシュボードから、メールアドレスまたは既に W&B アカウントを持つ場合は W&B のユーザー名で招待できます。

1. `https://wandb.ai/<team-name>` に移動します。
2. ダッシュボード左側のグローバルナビゲーションで **Team settings** を選択します。
{{< img src="/images/hosting/team_settings.png" alt="Team settings" >}}
3. **Users** タブを選択します。
4. **Invite a new user** を選択します。
5. 表示されるモーダルの **Email or username** フィールドにその user のメールアドレスを入力し、**Select a team role** ドロップダウンからその user に割り当てるロールを選択します。team 内で user が持てるロールの詳細は [Team roles]({{< relref path="#assign-or-update-a-team-members-role" lang="ja" >}}) を参照してください。
6. **Send invite** ボタンを選択します。

デフォルトでは、team 管理者またはインスタンス管理者のみが team にメンバーを招待できます。この振る舞いを変更するには、[Team settings]({{< relref path="/guides/models/app/settings-page/team-settings.md#privacy" lang="ja" >}}) を参照してください。

メール招待での手動追加に加え、新規 user の [メールアドレスが組織のドメインに一致する場合]({{< relref path="#domain-capture" lang="ja" >}})、その user を team に自動追加できます。

### サインアップ時に組織の team を見つけやすくする

サインアップ時に、組織内の新規 Users が組織の Teams を見つけられるようにします。新規 Users のメールドメインは、組織で検証済みのメールドメインと一致している必要があります。検証済みの新規 Users は、W&B アカウント作成時にその組織に属する検証済み Teams の一覧を表示できます。

この機能には組織のドメイン取得を有効化する必要があります。有効化手順は [ドメインキャプチャ]({{< relref path="#domain-capture" lang="ja" >}}) を参照してください。


### チームメンバーのロールを割り当て・更新する


1. メンバー名の横にあるアカウントタイプのアイコンを選択します。
2. ドロップダウンから、そのメンバーに付与したいアカウントタイプを選択します。

以下は team のメンバーに割り当て可能なロールです:

| Role | 定義 |
|-----------|---------------------------|
| Admin | team 内の他の Users を追加・削除し、user ロールを変更し、team の設定を構成できます。 |
| Member | team 管理者によってメールまたは組織レベルのユーザー名で招待された、team の一般 user。Member は他の Users を team に招待できません。 |
| View-Only (Enterprise のみ) | team 管理者によってメールまたは組織レベルのユーザー名で招待された、team の閲覧専用 user。team とそのコンテンツに対して読み取り専用アクセスのみを持ちます。 |
| Custom Roles (Enterprise のみ) | カスタムロールにより、組織管理者は View-Only または Member を継承して新しいロールを作成し、権限を追加してきめ細かなアクセス制御を実現できます。チーム管理者は、それらのカスタムロールを自身のチームの Users に割り当てられます。詳細は [custom roles のアナウンス](https://wandb.ai/wandb_fc/announcements/reports/Introducing-Custom-Roles-for-W-B-Teams--Vmlldzo2MTMxMjQ3) を参照してください。 |

{{% alert color="info" %}}
サービスアカウントは Users ではなく、自動化のために使用する非人間の ID です。**Service accounts** は自動化されたワークフロー向けの API キーを提供し、user ライセンスを消費しません。サービスアカウントの作成と管理については、[Service accounts でワークフローを自動化する]({{< relref path="/guides/hosting/iam/authentication/service-accounts.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

{{% alert %}}
専用クラウドまたは Self-managed デプロイメントのエンタープライズライセンスのみ、team メンバーにカスタムロールを割り当てられます。
{{% /alert %}}

### team から Users を削除する
team のダッシュボードから user を削除します。Runs の作成者が team を離れても、その team で作成された Runs は W&B に保持されます。

1. `https://wandb.ai/<team-name>` に移動します。
2. 左側のナビゲーションバーで **Team settings** を選択します。
3. **Users** タブを選択します。
4. 削除したい user 名の横にマウスポインタを合わせます。表示される三点リーダーアイコン (**...**) を選択します。
5. ドロップダウンから **Remove user** を選択します。