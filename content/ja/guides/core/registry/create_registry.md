---
title: カスタムRegistryを作成する
menu:
  default:
    identifier: ja-guides-core-registry-create_registry
    parent: registry
weight: 2
---

カスタムRegistryを使うことで、利用可能な artifact の種類を柔軟に制御でき、Registryの公開範囲の制限なども行えます。

{{% pageinfo color="info" %}}
コアRegistryとカスタムRegistryの違いについては、[Registryタイプのまとめ]({{< relref path="registry_types.md#summary" lang="ja" >}})の表をご覧ください。
{{% /pageinfo %}}


## カスタムRegistryの作成

カスタムRegistryを作成するには:

1. https://wandb.ai/registry/ の **Registry** アプリにアクセスします。
2. **Custom registry** 内で、**Create registry** ボタンをクリックします。
3. **Name** フィールドに、Registryの名前を入力します。
4. 必要に応じて、Registryの説明を入力します。
5. **Registry visibility** ドロップダウンから、Registryを閲覧できるユーザーを選びます。Registryの公開範囲については[Registry公開範囲の種類]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ja" >}})をご参照ください。
6. **Accepted artifacts type** ドロップダウンから、**All types** または **Specify types** を選択します。
7. （**Specify types** を選択した場合）Registryで受け入れる artifact の種類を1つ以上追加します。
8. **Create registry** ボタンをクリックします。

{{% alert %}}
artifact の種類は、一度Registryの設定に保存すると後から削除できません。
{{% /alert %}}

例えば、次の画像は、`Fine_Tuned_Models` というカスタムRegistryを作成しようとしている例です。このRegistryは**Restricted**（制限付き）で、手動で追加されたメンバーだけが利用できます。

{{< img src="/images/registry/create_registry.gif" alt="Creating a new registry" >}}

## 公開範囲の種類

Registryの *公開範囲* は、そのRegistryに誰がアクセスできるかを決定します。公開範囲を制限することで、特定のメンバーのみがアクセスできるようにできます。

カスタムRegistryの公開範囲オプションは2つあります:

| 公開範囲 | 説明 |
| --- | --- | 
| Restricted（制限付き）   | 招待された組織メンバーのみがRegistryにアクセス可能です。| 
| Organization（組織全体） | 組織内の全員がRegistryにアクセスできます。 |

チーム管理者またはRegistry管理者が、カスタムRegistryの公開範囲を設定できます。

**Restricted** な公開範囲でカスタムRegistryを作成した場合、作成者は自動的にそのRegistryの管理者として追加されます。


## カスタムRegistryの公開範囲を設定する

チーム管理者またはRegistry管理者は、カスタムRegistryの作成時や作成後に、その公開範囲を指定できます。

既存のカスタムRegistryの公開範囲を制限するには:

1. https://wandb.ai/registry/ の **Registry** アプリにアクセスします。
2. 対象のRegistryを選択します。
3. 右上の歯車アイコンをクリックします。
4. **Registry visibility** ドロップダウンから、希望する公開範囲を選択します。
5. **Restricted visibility** を選択した場合は、以下の手順に従います:
   1. Registryにアクセスさせたい組織メンバーを追加します。**Registry members and roles** セクションまでスクロールし、**Add member** ボタンをクリックします。
   2. **Member** フィールドに、追加するメンバーのメールアドレスまたはユーザー名を入力します。
   3. **Add new member** をクリックします。

{{< img src="/images/registry/change_registry_visibility.gif" alt="Changing registry visibility settings from private to public or team-restricted access" >}}

チーム管理者がカスタムRegistryを作成する際の公開範囲の設定方法については、[カスタムRegistryの作成]({{< relref path="./create_registry.md#create-a-custom-registry" lang="ja" >}})をご覧ください。