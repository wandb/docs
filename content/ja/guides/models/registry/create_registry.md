---
title: Create a custom registry
menu:
  default:
    identifier: ja-guides-models-registry-create_registry
    parent: registry
weight: 2
---

カスタムレジストリは、使用できる artifact のタイプに対する柔軟性と制御を提供し、レジストリの可視性を制限できます。

{{% pageinfo color="info" %}}
コアレジストリとカスタムレジストリの完全な比較については、[レジストリタイプ]({{< relref path="registry_types.md#summary" lang="ja" >}}) の概要表を参照してください。
{{% /pageinfo %}}

## カスタムレジストリの作成

カスタムレジストリを作成するには:

1. https://wandb.ai/registry/ の **Registry** App に移動します。
2. **Custom registry** 内で、**Create registry** ボタンをクリックします。
3. **Name** フィールドにレジストリの名前を入力します。
4. 必要に応じて、レジストリに関する説明を入力します。
5. **Registry visibility** ドロップダウンから、レジストリを表示できるユーザーを選択します。レジストリの可視性オプションの詳細については、[レジストリの可視性タイプ]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ja" >}}) を参照してください。
6. **Accepted artifacts type** ドロップダウンから、**All types** または **Specify types** のいずれかを選択します。
7. (**Specify types** を選択した場合) レジストリが受け入れる 1 つ以上の artifact タイプを追加します。
8. **Create registry** ボタンをクリックします。

{{% alert %}}
artifact タイプは、レジストリの設定に保存された後、レジストリから削除できません。
{{% /alert %}}

たとえば、次の図は、 ユーザー が作成しようとしている `Fine_Tuned_Models` という名前のカスタムレジストリを示しています。レジストリは、手動でレジストリに追加されたメンバーのみに **Restricted** されています。

{{< img src="/images/registry/create_registry.gif" alt="" >}}

## 可視性タイプ

レジストリの _可視性_ は、誰がそのレジストリにアクセスできるかを決定します。カスタムレジストリの可視性を制限すると、指定されたメンバーのみがそのレジストリにアクセスできるようになります。

カスタムレジストリには、次の 2 種類のレジストリ可視性オプションがあります。

| 可視性 | 説明 |
| --- | --- |
| Restricted | 招待された organization メンバーのみがレジストリにアクセスできます。 |
| Organization | organization 内のすべての ユーザー がレジストリにアクセスできます。 |

チーム管理者またはレジストリ管理者は、カスタムレジストリの可視性を設定できます。

Restricted の可視性でカスタムレジストリを作成した ユーザー は、レジストリ管理者として自動的にレジストリに追加されます。

## カスタムレジストリの可視性の設定

チーム管理者またはレジストリ管理者は、カスタムレジストリの作成中または作成後に、カスタムレジストリの可視性を割り当てることができます。

既存のカスタムレジストリの可視性を制限するには:

1. https://wandb.ai/registry/ の **Registry** App に移動します。
2. レジストリを選択します。
3. 右上隅にある歯車アイコンをクリックします。
4. **Registry visibility** ドロップダウンから、目的のレジストリの可視性オプションを選択します。
5. **Restricted visibility** を選択した場合:
   1. このレジストリへのアクセスを許可する organization のメンバーを追加します。**Registry members and roles** セクションまでスクロールし、**Add member** ボタンをクリックします。
   2. **Member** フィールドに、追加するメンバーのメールアドレスまたは ユーザー 名を追加します。
   3. **Add new member** をクリックします。

{{< img src="/images/registry/change_registry_visibility.gif" alt="" >}}

チーム管理者がカスタムレジストリを作成するときに、カスタムレジストリの可視性を割り当てる方法の詳細については、[カスタムレジストリの作成]({{< relref path="./create_registry.md#create-a-custom-registry" lang="ja" >}}) を参照してください。
