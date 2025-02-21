---
title: Migrate from legacy Model Registry
menu:
  default:
    identifier: ja-guides-models-registry-model_registry_eol
    parent: registry
weight: 8
---

W&B は、従来の [W&B Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) から新しい [W&B Registry]({{< relref path="./" lang="ja" >}}) へとアセットを移行します。この移行は完全に W&B によって管理され、トリガーされるため、 ユーザー からの介入は必要ありません。この プロセス は、既存の ワークフロー への混乱を最小限に抑え、可能な限りシームレスになるように設計されています。

移行は、新しい W&B Registry が Model Registry で現在利用可能なすべての機能を含むと行われます。W&B は、現在の ワークフロー 、 コード ベース、および参照を維持しようとします。

この ガイド は生きたドキュメントであり、より多くの情報が利用可能になり次第、定期的に更新されます。ご質問やサポートについては、support@wandb.com までご連絡ください。

## W&B Registry と従来の Model Registry との違い

W&B Registry は、 モデル 、 データセット 、およびその他の アーティファクト を管理するための、より堅牢で柔軟な 環境 を提供するために設計された、さまざまな新機能と機能拡張を導入します。

{{% alert %}}
従来の Model Registry を表示するには、W&B アプリ の Model Registry に移動します。ページの上部にバナーが表示され、従来の Model Registry アプリ UI を使用できます。

{{< img src="/images/registry/nav_to_old_model_reg.gif" >}}
{{% /alert %}}

### 組織の可視性
従来の Model Registry にリンクされた Artifacts は、 チーム レベルの可視性を持ちます。これは、 チーム の メンバー のみが、従来の W&B Model Registry で Artifacts を表示できることを意味します。W&B Registry は、組織レベルの可視性を持ちます。これは、適切な権限を持つ組織全体の メンバー が、 Registry にリンクされた Artifacts を表示できることを意味します。

### Registry への可視性を制限する
カスタム Registry を表示および アクセス できる ユーザー を制限します。カスタム Registry を作成するとき、またはカスタム Registry の作成後に、 Registry への可視性を制限できます。制限された Registry では、選択された メンバー のみがコンテンツに アクセス でき、プライバシーと コントロール が維持されます。Registry の可視性の詳細については、[Registry の可視性タイプ]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ja" >}})を参照してください。

### カスタム Registry の作成
従来の Model Registry とは異なり、W&B Registry は モデル や データセット の Registry に限定されません。特定の ワークフロー や プロジェクト のニーズに合わせて調整されたカスタム Registry を作成でき、任意の オブジェクト タイプを保持できます。この柔軟性により、 チーム は独自の要件に従って Artifacts を整理および管理できます。カスタム Registry の作成方法の詳細については、[カスタム Registry の作成]({{< relref path="./create_registry.md" lang="ja" >}})を参照してください。

{{< img src="/images/registry/mode_reg_eol.png" alt="" >}}

### カスタム アクセス コントロール
各 Registry は詳細な アクセス コントロール をサポートしており、 メンバー には管理者、 メンバー 、または閲覧者などの特定のロールを割り当てることができます。管理者は、 メンバー の追加または削除、ロールの設定、可視性の構成など、 Registry の 設定 を管理できます。これにより、 チーム は Registry 内の Artifacts を表示、管理、および操作できる ユーザー を必要な コントロール で管理できます。

{{< img src="/images/registry/registry_access_control.png" alt="" >}}

### 用語の更新
登録済み モデル は、「コレクション」と呼ばれるようになりました。

### 変更の概要

|               | 従来の W&B Model Registry | W&B Registry |
| -----         | ----- | ----- |
| Artifact の可視性| チーム の メンバー のみが Artifacts を表示または アクセス できます | 組織内の メンバー は、適切な権限を持っていれば、 Registry にリンクされた Artifacts を表示または アクセス できます |
| カスタム アクセス コントロール | 利用不可 | 利用可能 |
| カスタム Registry | 利用不可 | 利用可能 |
| 用語の更新 | モデル バージョンへの一連のポインター（リンク）は、「登録済み モデル 」と呼ばれます。 | アーティファクト バージョンへの一連のポインター（リンク）は、「コレクション」と呼ばれます。 |
| `wandb.init.link_model` | Model Registry 固有の API | 現在は従来の Model Registry とのみ互換性があります |

## 移行の準備

W&B は、登録済みの モデル (現在はコレクションと呼ばれています) と、従来の Model Registry から W&B Registry に関連付けられた アーティファクト バージョンを移行します。この プロセス は自動的に実行され、 ユーザー からの操作は必要ありません。

### チーム の可視性から組織の可視性へ

移行後、 Model Registry は組織レベルの可視性を持つようになります。[ロールを割り当てる]({{< relref path="./configure_registry.md" lang="ja" >}})ことで、 Registry への アクセス 権を持つ ユーザー を制限できます。これにより、特定の メンバー のみが特定の Registry に アクセス できるようになります。

移行により、従来の W&B Model Registry での現在の チーム レベルの登録済み モデル (まもなくコレクションと呼ばれるようになります) の既存のアクセス許可の境界が維持されます。従来の Model Registry で現在定義されているアクセス許可は、新しい Registry でも保持されます。これは、現在特定の チーム メンバー に制限されているコレクションが、移行中および移行後も保護されたままになることを意味します。

### Artifact のパスの継続性

現在、必要な操作はありません。

## 移行中

W&B は移行 プロセス を開始します。移行は、W&B サービスの混乱を最小限に抑える時間枠内で行われます。従来の Model Registry は、移行が開始されると読み取り専用状態に移行し、参照用に アクセス 可能なままになります。

## 移行後

移行後、コレクション、 アーティファクト バージョン、および関連する属性は、新しい W&B Registry 内で完全に アクセス 可能になります。現在の ワークフロー をそのまま維持することに重点を置き、変更をナビゲートするための継続的なサポートを提供します。

### 新しい Registry の使用

ユーザー は、W&B Registry で利用できる新機能と能力を検討することをお勧めします。Registry は、現在依存している機能だけでなく、カスタム Registry 、可視性の向上、柔軟な アクセス コントロール などの機能拡張も導入します。

W&B Registry を早期に試してみたい場合、または Registry から開始したい新しい ユーザー 向けのサポートが利用可能であり、従来の W&B Model Registry は利用できません。この機能を有効にするには、support@wandb.com またはセールス MLE にお問い合わせください。早期移行は BETA バージョンになることに注意してください。BETA バージョンの W&B Registry には、従来の Model Registry のすべての機能または特徴がない場合があります。

詳細と W&B Registry の全機能の詳細については、[W&B Registry ガイド]({{< relref path="./" lang="ja" >}})をご覧ください。

## よくある質問

#### W&B が Model Registry から W&B Registry に アセット を移行するのはなぜですか？

W&B は、新しい Registry でより高度な機能と能力を提供するために、 プラットフォーム を進化させています。この移行は、 モデル 、 データセット 、およびその他の アーティファクト を管理するための、より統合された強力な ツール セットを提供する上で重要なステップです。

#### 移行前に何をする必要がありますか？

移行前に ユーザー からの操作は必要ありません。W&B が移行を処理し、 ワークフロー と参照が保持されるようにします。

#### モデル アーティファクト への アクセス 権は失われますか？

いいえ、 モデル アーティファクト への アクセス 権は、移行後も保持されます。従来の Model Registry は読み取り専用状態のままであり、すべての関連 データ が新しい Registry に移行されます。

#### Artifacts に関連する メタデータ は保持されますか？

はい、 Artifacts の作成、 リネージ 、およびその他の属性に関連する重要な メタデータ は、移行中に保持されます。ユーザー は移行後もすべての関連 メタデータ に アクセス し続けるため、 Artifacts の整合性とトレーサビリティが維持されます。

#### サポートが必要な場合は、誰に連絡すればよいですか？

ご質問やご不明な点がございましたら、サポートをご利用いただけます。support@wandb.com までお問い合わせください。
