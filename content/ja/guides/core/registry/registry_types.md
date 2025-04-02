---
title: Registry types
menu:
  default:
    identifier: ja-guides-core-registry-registry_types
    parent: registry
weight: 1
---

W&B は、[コアレジストリ]({{< relref path="#core-registry" lang="ja" >}}) と [カスタムレジストリ]({{< relref path="#custom-registry" lang="ja" >}}) の2種類のレジストリをサポートしています。

## コアレジストリ
コアレジストリは、特定の ユースケース ( **Models** と **Datasets** ) のためのテンプレートです。

デフォルトでは、**Models** レジストリは `"model"` artifact タイプを受け入れるように設定され、**Dataset** レジストリは `"dataset"` artifact タイプを受け入れるように設定されています。管理者は、追加の受け入れ可能な artifact タイプを追加できます。

{{< img src="/images/registry/core_registry_example.png" alt="" >}}

上の図は、W&B Registry App UI の **Models** と **Dataset** コアレジストリと、**Fine_Tuned_Models** というカスタムレジストリを示しています。

コアレジストリは、[組織の可視性]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ja" >}}) を持っています。レジストリ管理者は、コアレジストリの可視性を変更できません。

## カスタムレジストリ
カスタムレジストリは、`"model"` artifact タイプまたは `"dataset"` artifact タイプに制限されません。

初期の データ 収集から最終的な モデル の デプロイメント まで、機械学習 パイプライン の各ステップに対してカスタムレジストリを作成できます。

たとえば、トレーニング された モデル のパフォーマンスを評価するために、キュレーションされた データセット を整理するために「Benchmark_Datasets」というレジストリを作成できます。このレジストリ内には、「User_Query_Insurance_Answer_Test_Data」というコレクションがあり、トレーニング 中に モデル が見たことのない ユーザー の質問と、対応する専門家によって検証された回答のセットが含まれている場合があります。

{{< img src="/images/registry/custom_registry_example.png" alt="" >}}

カスタムレジストリは、[組織または制限付きの可視性]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ja" >}}) を持つことができます。レジストリ管理者は、カスタムレジストリの可視性を組織から制限付きに変更できます。ただし、レジストリ管理者は、カスタムレジストリの可視性を制限付きから組織の可視性には変更できません。

カスタムレジストリの作成方法については、[カスタムレジストリの作成]({{< relref path="./create_collection.md" lang="ja" >}}) を参照してください。

## まとめ
次の表は、コアレジストリとカスタムレジストリの違いをまとめたものです。

|                | Core  | Custom|
| -------------- | ----- | ----- |
| 可視性     | 組織の可視性のみ。可視性は変更できません。 | 組織または制限付き。可視性は組織から制限付きの可視性へ変更できます。|
| メタデータ       | 事前設定されており、 ユーザー は編集できません。 | ユーザー が編集できます。  |
| Artifact タイプ | 事前設定されており、受け入れられた artifact タイプは削除できません。 ユーザー は追加の受け入れられた artifact タイプを追加できます。 | 管理者は、受け入れられたタイプを定義できます。 |
| カスタマイズ    | 既存のリストにタイプを追加できます。|  レジストリ名、説明、可視性、および受け入れられた artifact タイプを編集します。|
