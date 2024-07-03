---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create a collection

レジストリ内にコレクションを作成し、アーティファクトを整理します。*コレクション* とは、レジストリ内でリンクされたアーティファクトバージョンのセットを指します。各コレクションは、特定のタスクやユースケースを表し、そのタスクに関連する選定されたアーティファクトバージョンのコンテナとして機能します。

![](/images/registry/what_is_collection.png)

例えば、上の画像は「Forecast」という名前のレジストリを示しています。「Forecast」レジストリ内には、「LowLightPedRecog-YOLO」と「TextCat」という2つのコレクションがあります。

:::tip
W&B Model Registry に詳しい方は、「registered models」をご存知かもしれません。W&B Registry では、registered models が「collections」に名前が変更されています。 [モデルレジストリでregistered modelsを作成する](../model_registry/create-registered-model.md) 方法は、W&B Registryでコレクションを作成する方法とほぼ同じです。主な違いは、コレクションがregistered modelsのようにエンティティに属さない点です。
:::

以下の手順は、W&B Registry App UIを使用してレジストリ内にコレクションを作成する方法を示します：

1. W&B App UIで **Registry** App に移動します。
2. レジストリを選択します。
3. 右上隅の **Create collection** ボタンをクリックします。
4. **Name** フィールドにコレクションの名前を入力します。
5. **Type** ドロップダウンからタイプを選択します。または、レジストリがカスタムアーティファクトタイプを許可している場合、このコレクションが受け入れる1つ以上のアーティファクトタイプを提供します。
:::info
アーティファクトタイプは一度レジストリに追加され保存されると、レジストリの設定から削除することはできません。
:::
6. 必要に応じて **Description** フィールドにコレクションの説明を入力します。
7. 必要に応じて **Tags** フィールドに1つ以上のタグを追加します。
8. **Link version** をクリックします。
9. **Project** ドロップダウンから、アーティファクトが保存されているプロジェクトを選択します。
10. **Artifact** コレクションのドロップダウンから、アーティファクトを選択します。
11. **Version** ドロップダウンから、コレクションにリンクするアーティファクトバージョンを選択します。
12. **Create collection** ボタンをクリックします。

![](/images/registry/create_collection.gif)