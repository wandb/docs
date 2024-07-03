---
displayed_sidebar: default
---

# カスタムレジストリを作成する

各ステップのMLワークフローにおいて、[カスタムレジストリ](./registry_types.md#custom-registry)を作成します。

カスタムレジストリは、デフォルトの[コアレジストリ](./registry_types.md#core-registry)とは異なるプロジェクト固有の要件を整理するのに特に便利です。

次の手順では、対話的にレジストリを作成する方法について説明します:
1. W&BアプリUIの**Registry**アプリに移動します。
2. **Custom registry**内で、**Create registry**ボタンをクリックします。
3. **Name**フィールドにレジストリの名前を入力します。
4. 必要に応じて、レジストリについての説明を提供します。
5. **Registry visibility**ドロップダウンからレジストリを表示できる人を選択します。レジストリの表示オプションについての詳細は[Registry visibility types](./configure_registry.md#registry-visibility-types)をご覧ください。
6. **Accepted artifacts type**ドロップダウンから**All types**または**Specify types**を選択します。
7. (**Specify types**を選択した場合) レジストリが受け入れるアーティファクトタイプを1つ以上追加します。
:::info
アーティファクトタイプは、一度レジストリに追加して保存されると、レジストリの設定から削除することはできません。
:::
8. **Create registry**ボタンをクリックします。



![](/images/registry/create_registry.gif)

例えば、上の画像は「Fine_Tuned_Models」というカスタムレジストリをユーザーが作成しようとしているところを示しています。このレジストリは**Restricted**に設定されており、この「Fine_Tuned_Models」レジストリに手動で追加されたメンバーのみがこのレジストリにアクセスできます。