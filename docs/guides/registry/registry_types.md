---
displayed_sidebar: default
---

# Registry types

W&B は 2 種類のレジストリをサポートしています：[Core registries](#core-registry) と [Custom registries](#custom-registry)。

## Core registry

Core registry は特定のユースケースのテンプレートです: **Models** と **Datasets**。

デフォルトでは、**Models** レジストリは `"model"` アーティファクトタイプを受け入れるように設定されており、**Dataset** レジストリは `"dataset"` アーティファクトタイプを受け入れるように設定されています。管理者は追加のアーティファクトタイプを受け入れ可能にすることができます。

![](/images/registry/core_registry_example.png)

上の画像は W&B Registry アプリの UI で表示される **Models** と **Dataset** のコアレジストリに加え、**Fine_Tuned_Models** というカスタムレジストリを示しています。

Core registry は [organization visibility](./configure_registry.md#registry-visibility-types) を持ちます。レジストリの管理者は core registry の visibility を変更することはできません。

## Custom registry

Custom registries は `"model"` アーティファクトタイプや `"dataset"` アーティファクトタイプに制限されません。

カスタムレジストリは、初期データ収集から最終的なモデルデプロイメントまで、機械学習パイプラインの各ステップのために作成することができます。

例えば、訓練されたモデルの性能を評価するためにキュレートされたデータセットを整理するための "Benchmark_Datasets" というレジストリを作成することができます。このレジストリ内には、トレーニング中にモデルが一度も見たことのない、ユーザーの質問と対応する専門家によって検証された回答のセットを含む "User_Query_Insurance_Answer_Test_Data" というコレクションが含まれるかもしれません。

![](/images/registry/custom_registry_example.png)

Custom registry は [organization または restricted visibility](./configure_registry.md#registry-visibility-types) を持つことができます。レジストリの管理者はカスタムレジストリの visibility を組織から制限付きに変更することができます。ただし、カスタムレジストリの visibility を制限付きから組織へ変更することはできません。

カスタムレジストリの作成方法については、[Create a custom registry](./create_collection.md) を参照してください。

## Summary

以下の表は、Core と Custom のレジストリの違いをまとめています:

|                | Core  | Custom|
| -------------- | ----- | ----- |
| Visibility     | Organizational visibility のみ。visibility を変更できない。 | Organization または restricted のどちらか。visibility を組織から制限付き visibility に変更できる。|
| Metadata       | 事前設定されており、ユーザーによる編集はできない。 | ユーザーが編集可能。  |
| Artifact types | 事前設定されており、受け入れられるアーティファクトタイプを削除できない。ユーザーは追加のアーティファクトタイプを追加できる。 | 管理者が受け入れられるタイプを定義できる。 |
| Customization  | 既存のリストに追加のタイプを追加可能。 | レジストリの名前、説明、visibility、および受け入れられるアーティファクトタイプを編集できる。 |