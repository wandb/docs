---
title: レガシー Registered Models からの移行
menu:
  default:
    identifier: ja-guides-core-registry-model_registry_eol
    parent: registry
weight: 9
---

W&B は、旧 **Model Registry** から強化された **W&B Registry** への移行を進めています。この移行は、W&B がシームレスかつ完全に管理するように設計されています。移行プロセスでは、ワークフローを維持しつつ、強力な新機能を利用できるようになります。ご質問やサポートについては、[support@wandb.com](mailto:support@wandb.com) までお問い合わせください。

## 移行の理由

W&B Registry は、旧 Model Registry と比較して大幅な改善が提供されます。

- **統一された組織レベルのエクスペリエンス**: チームに関係なく、組織全体で厳選された Artifacts を共有および管理できます。
- **ガバナンスの向上**: アクセス制御、制限付きレジストリ、公開範囲の設定を使用して、ユーザーのアクセスを管理できます。
- **機能の強化**: カスタムレジストリ、より優れた検索、監査証跡、オートメーションのサポートなどの新機能は、ML インフラストラクチャーの最新化に役立ちます。

以下の表は、旧 Model Registry と新しい W&B Registry の主な違いをまとめたものです。

| 機能 | 旧 W&B Model Registry | W&B Registry |
| ----- | ----- | ----- |
| Artifacts の公開範囲 | チームレベルのみ — チームメンバーにアクセスが制限 | 組織レベルの公開範囲 — 詳細な権限制御が可能 |
| カスタムレジストリ | サポート対象外 | 完全にサポート — あらゆる Artifacts タイプに対してレジストリを作成可能 |
| アクセス制御 | 利用不可 | レジストリレベルでのロールベースアクセス (管理者、メンバー、閲覧者) |
| 用語 | 「Registered models」：Model のバージョンへのポインター | 「Collections」：任意の Artifacts バージョンへのポインター |
| レジストリのスコープ | Model のバージョン管理のみをサポート | Models、Datasets、カスタム Artifacts などに対応 |
| オートメーション | レジストリレベルのオートメーション | レジストリおよび Collection レベルのオートメーションがサポートされ、移行中にコピーされます |
| 検索と検出 | 検索と検出が限定的 | W&B Registry 内で組織内のすべてのレジストリを横断する中央検索 |
| API の互換性 | `wandb.init.link_model()` と MR 固有のパターンを使用 | 自動リダイレクト機能付きの最新 SDK API (`link_artifact()`, `use_artifact()`) |
| 移行 | サポート終了 | 自動的に移行および強化 — データはコピーされ、削除されません |

## 移行の準備

- **アクションは不要**: 移行は W&B によって完全に自動化され、管理されます。スクリプトを実行したり、設定を更新したり、データを手動で移動したりする必要はありません。
- **情報をご確認ください**: 予定されている移行の 2 週間前に、W&B App UI のバナーで通知を受け取ります。
- **権限の確認**: 移行後、管理者はチームのニーズに合わせてレジストリのアクセス権を確認する必要があります。
- **今後の作業では新しいパスを使用**: 古いコードは引き続き動作しますが、W&B は新しい Projects には新しい W&B Registry パスを使用することをお勧めします。

## 移行プロセス

### 一時的な書き込み操作の一時停止
移行中、チームの Model Registry への書き込み操作は、データの整合性を確保するため、最大 1 時間一時停止されます。新しく作成された移行済みの W&B Registry への書き込み操作も、移行中に一時停止されます。

### データ移行
W&B は、旧 Model Registry から新しい W&B Registry へ以下のデータを移行します。

- Collections
- リンクされた Artifacts のバージョン
- バージョン履歴
- エイリアス、タグ、説明
- オートメーション (Collection およびレジストリレベルの両方)
- 権限 (サービスアカウントのロールおよび保護されたエイリアスを含む)

W&B App UI 内では、旧 Model Registry は新しい W&B Registry に置き換えられます。移行されたレジストリには、チーム名に続いて `mr-migrated` が付加されます。

```text
<team-name>-mr-migrated
```

これらのレジストリはデフォルトで**制限付き**の公開範囲となり、既存のプライバシー境界を維持します。`<team-name>` の元のメンバーのみが、それぞれのレジストリにアクセスできます。

## 移行後

移行が完了すると、次のようになります。

- 旧 Model Registry は**読み取り専用**になります。引き続きデータを表示およびアクセスできますが、新規の書き込みは許可されません。
- 旧 Model Registry のデータは、新しい W&B Registry に**コピー**され、移動されません。データは削除されません。
- 新しい W&B Registry からすべてのデータにアクセスします。
- バージョン管理、ガバナンス、監査証跡、オートメーションには、新しい Registry UI を使用します。
- 古いコードは引き続き使用できます。
   - [既存のパスと API 呼び出しは、自動的に新しい W&B Registry にリダイレクトされます。]({{< relref path="#code-will-continue-to-work" lang="ja" >}})
   - [Artifacts のバージョンパスはリダイレクトされます。]({{< relref path="#legacy-paths-will-redirect-to-new-wb-registry-paths" lang="ja" >}})
- 旧 Model Registry は一時的に UI に表示され続けます。W&B は最終的に旧 Model Registry を非表示にします。
- Registry の強化された機能を確認できます。例えば、以下のような機能があります。
    - [組織レベルのアクセス]({{< relref path="/guides/core/registry/create_registry/#visibility-types" lang="ja" >}})
    - [ロールベースのアクセス制御]({{< relref path="/guides/core/registry/configure_registry/" lang="ja" >}})
    - [レジストリレベルのリネージトラッキング]({{< relref path="/guides/core/registry/lineage/" lang="ja" >}})
    - [オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}})

### コードは引き続き動作します

コード内で旧 Model Registry を参照している既存の API 呼び出しは、自動的に新しい W&B Registry にリダイレクトされます。以下の API 呼び出しは、変更なしで引き続き動作します。

- `wandb.Api().artifact()`
- `wandb.run.use_artifact()`
- `wandb.run.link_artifact()`
- `wandb.Artifact().link()`

### 旧パスは新しい W&B Registry パスにリダイレクトされます

W&B は、旧 Model Registry のパスを新しい W&B Registry 形式に自動的にリダイレクトします。これは、パスをすぐにリファクタリングすることなく、既存のコードを継続して使用できることを意味します。自動リダイレクトは、移行前に旧 Model Registry で作成された Collection にのみ適用されることに注意してください。

例:
- 旧 Model Registry に Collection `"my-model"` がすでに存在する場合、リンクアクションは正常にリダイレクトされます
- 旧 Model Registry に Collection `"my-model"` が存在しなかった場合、リダイレクトされず、エラーが発生します

```python
# 旧 Model Registry に "my-model" が存在していれば、正常にリダイレクトされます
run.link_artifact(artifact, "team-name/model-registry/my-model")

# 旧 Model Registry に "new-model" が存在しなかった場合は失敗します
run.link_artifact(artifact, "team-name/model-registry/new-model")
```

旧 Model Registry からバージョンを取得するためのパスは、チーム名、`「model-registry」` という文字列、Collection 名、およびバージョンで構成されていました。

```python
f"{team-name}/model-registry/{collection-name}:{version}"
```

W&B はこれらのパスを新しい W&B Registry 形式に自動的にリダイレクトします。これには、組織名、`「wandb-registry」` という文字列、チーム名、Collection 名、およびバージョンが含まれます。

```python
# 新しいパスへリダイレクトされます
f"{org-name}/wandb-registry-{team-name}/{collection-name}:{version}"
```

{{% alert title="Python SDK の警告" %}}

コードで旧 Model Registry パスを使い続けると、警告エラーが表示されることがあります。この警告によってコードが動作しなくなることはありませんが、パスを新しい W&B Registry 形式に更新する必要があることを示しています。

警告が表示されるかどうかは、使用している W&B Python SDK のバージョンによって異なります。

* 最新の W&B SDK (`v0.21.0` 以上) を使用しているユーザーには、リダイレクトが発生したことを示す、コードを中断しない警告がログに表示されます。
* 古い SDK バージョンの場合、リダイレクトは警告を発することなくサイレントに動作します。エンティティ名や Projects 名などの一部のメタデータは、古い値を反映している可能性があります。

{{% /alert %}}

## よくある質問

### 組織がいつ移行されるかはどのようにわかりますか？

アプリ内バナーまたは W&B からの直接の連絡により、事前に通知を受け取ります。

### ダウンタイムは発生しますか？

移行中、旧 Model Registry と新しい W&B Registry への書き込み操作は、約 1 時間一時停止されます。他のすべての W&B サービスは引き続き利用可能です。

### これによりコードが壊れますか？

いいえ。すべての旧 Model Registry パスと Python SDK 呼び出しは、自動的に新しい Registry にリダイレクトされます。

### データは削除されますか？

いいえ。データは新しい W&B Registry にコピーされます。旧 Model Registry は読み取り専用になり、後で非表示になります。データが削除されたり失われたりすることはありません。

### 古い SDK を使用している場合はどうなりますか？

リダイレクトは引き続き動作しますが、それに関する警告は表示されません。最適なエクスペリエンスのために、W&B SDK の最新バージョンにアップグレードしてください。

### 移行されたレジストリの名前を変更/修正できますか？

はい、移行されたレジストリの名前変更や、メンバーの追加または削除などの操作は許可されています。これらのレジストリは単なるカスタムレジストリであり、移行後もリダイレクトは機能し続けます。

## ご質問がありますか？

サポートまたは移行に関するご相談は、[support@wandb.com](mailto:support@wandb.com) までお問い合わせください。W&B は、お客様が新しい W&B Registry へスムーズに移行できるよう尽力します。