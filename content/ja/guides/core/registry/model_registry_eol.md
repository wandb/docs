---
title: レガシーモデルRegistry からの移行
menu:
  default:
    identifier: ja-guides-core-registry-model_registry_eol
    parent: registry
weight: 9
---

W&B は、従来の **Model Registry** から強化された **W&B Registry** への移行を進めています。この移行はシームレスで、W&B がすべて管理します。移行プロセス中も既存のワークフローは維持され、さらに強力な新機能が利用できるようになります。ご質問やサポートが必要な場合は、[support@wandb.com](mailto:support@wandb.com) までご連絡ください。

## 移行の理由

W&B Registry は、従来の Model Registry と比べて大幅な改善が施されています。

- **統一された組織レベルの体験**: チームを問わず、組織全体で厳選されたアーティファクトを共有・管理可能です。
- **ガバナンスの強化**: アクセス制御や制限付きRegistry、公開範囲の設定などによるユーザー管理が可能です。
- **機能拡張**: カスタムRegistry、優れた検索、監査履歴、オートメーション対応など、ML インフラを最新化する機能を搭載。



下記の表は、従来の Model Registry と新しい W&B Registry の主な違いをまとめたものです。

| 機能 | 従来の W&B Model Registry | W&B Registry |
| ----- | ----- | ----- |
| アーティファクトの公開範囲 | チーム単位のみ - メンバー限定 | 組織単位で詳細な権限管理が可能 |
| カスタムRegistry | 非対応 | 完全対応 — どんなアーティファクトにもRegistryを作成可能 |
| アクセス制御 | 非対応 | Registry単位のロールベースアクセス（Admin、Member、Viewer）|
| 用語 | “Registered models”: モデルバージョンへのポインタ | “Collections”: 任意のアーティファクトバージョンへのポインタ |
| Registryの範囲 | モデルのバージョン管理のみ対応 | モデル、データセット、カスタムアーティファクトなど多様に対応 |
| オートメーション | Registry単位のみ対応 | Registry・コレクション単位両方対応、移行時コピーされます |
| 検索・発見性 | 検索・発見性が限定的 | 組織内すべてのRegistryを中央検索可能 |
| API 互換性 | `wandb.init.link_model()` や MR専用パターンを使用 | モダンな SDK API（`link_artifact()`, `use_artifact()`）で自動リダイレクト |
| 移行 | サポート終了 | 自動で移行・強化—データはコピー、削除されません |

## 移行準備

- **ユーザー操作は不要**: 移行は W&B が完全に自動化・管理します。スクリプト実行、設定変更、データの手動移動などは一切不要です。
- **連絡に注意**: 移行予定日の2週間前から W&B App UI 上でバナー等で通知が届きます。
- **権限の確認**: 移行後、管理者はRegistryアクセス権限がチームニーズに合っているかご確認ください。
- **今後の作業では新パスを利用**: 既存コードはそのまま動作しますが、新しいプロジェクトでは新しい W&B Registry パスの利用を推奨します。

## 移行プロセス

### 一時的な書き込み停止
移行作業中は、チームの Model Registry への書き込み操作が最大1時間一時停止されます。新たに作成される W&B Registry への書き込みも、移行中は一時的に停止されます。

### データ移行
W&B は、従来の Model Registry から新しい W&B Registry へ以下のデータを移行します。

- Collections
- リンクされたアーティファクトのバージョン
- バージョン履歴
- エイリアス、タグ、説明
- オートメーション（コレクション単位・Registry単位両方）
- 権限（サービスアカウントのロールや保護されたエイリアスも含む）

W&B App UI 上では、従来の Model Registry の表示が新しい W&B Registry に置き換わります。移行後のRegistry名は、チーム名の後ろに `mr-migrated` が付きます。

```text
<team-name>-mr-migrated
```

これらのRegistryはデフォルトで **Restricted** （制限付き）公開範囲となり、既存のプライバシー設定が維持されます。元の `<team-name>` メンバーのみが各自のRegistryへアクセス可能です。

## 移行後

移行完了後には以下の通りとなります：

- 従来の Model Registry は **読み取り専用** となります。データの閲覧や取得は可能ですが、新規書き込みはできません。
- 従来の Model Registry のデータは、新しい W&B Registry に **コピー** されるのみで、移動ではありません。データ削除はありません。
- すべてのデータは新しい W&B Registry からアクセスできます。
- バージョン管理、ガバナンス、監査履歴、オートメーションなどは新しい Registry UI から利用できます。
- 既存コードも継続して動作します。
   - [既存のパスや API コールは自動的に新しい W&B Registry へリダイレクトされます。]({{< relref path="#code-will-continue-to-work" lang="ja" >}})
   - [アーティファクトバージョンのパスもリダイレクトされます。]({{< relref path="#legacy-paths-will-redirect-to-new-wb-registry-paths" lang="ja" >}})
- 従来の Model Registry は一時的にUI上から見えますが、やがて非表示となります。
- Registry では、さらに強化された機能を試せます。例:
    - [組織単位のアクセ ス]({{< relref path="/guides/core/registry/create_registry/#visibility-types" lang="ja" >}})
    - [ロールベースのアクセス制御]({{< relref path="/guides/core/registry/configure_registry/" lang="ja" >}})
    - [Registry単位のリネージ追跡]({{< relref path="/guides/core/registry/lineage/" lang="ja" >}})
    - [オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}})

### コードはそのまま動作します

従来の Model Registry への参照を含む既存 API コールも自動的に新しい W&B Registry へリダイレクトされます。下記 API コールは変更なしで引き続き動作します。

- `wandb.Api().artifact()`
- `wandb.run.use_artifact()`
- `wandb.run.link_artifact()`
- `wandb.Artifact().link()`

### 従来パスも新しい W&B Registry パスにリダイレクトされます

W&B は、従来の Model Registry のパスを自動的に新しい W&B Registry フォーマットにリダイレクトします。これにより、既存コードをすぐにリファクタしなくても引き続き利用できます。ただし自動リダイレクトは、移行前に従来の Model Registry で作成されたコレクションにのみ適用されます。

例えば：
- 従来の Model Registry に `"my-model"` コレクションがすでに存在していれば、リンク操作は正常にリダイレクトされます
- `"my-model"` が存在しなければ、リダイレクトされずエラーとなります

```python
# "my-model" が従来の Model Registry に存在すれば正常にリダイレクト
run.link_artifact(artifact, "team-name/model-registry/my-model")

# "new-model" が従来の Model Registry に存在しなければ失敗
run.link_artifact(artifact, "team-name/model-registry/new-model")
```

従来の Model Registry からバージョンを取得する場合、パスはチーム名、`"model-registry"`、コレクション名、バージョンの順で構成されていました。

```python
f"{team-name}/model-registry/{collection-name}:{version}"
```

W&B はこれらのパスも自動で新しい W&B Registry 用フォーマットにリダイレクトします。新パスでは組織名、`"wandb-registry"`、チーム名、コレクション名、バージョンを含みます。

```python
# 新しいパスにリダイレクト
f"{org-name}/wandb-registry-{team-name}/{collection-name}:{version}"
```

{{% alert title="Python SDK の警告" %}}

従来の Model Registry パスを引き続き利用した場合、警告メッセージが表示される場合があります。警告が出てもコードは壊れませんが、新しい W&B Registry フォーマットにパスを更新することが推奨されています。

警告が表示されるかどうかは、利用中の W&B Python SDK のバージョンによります。

* 最新の W&B SDK（`v0.21.0` 以降）では、リダイレクト発生を示す警告ログが非破壊的に出力されます。
* 古い SDK バージョンでもリダイレクトは動作しますが、警告表示はされません。エンティティやプロジェクト名など一部メタデータは旧値を反映する場合があります。

{{% /alert %}} 

## よくある質問

### 移行時期はどのように通知されますか？

事前にアプリ内バナーもしくは W&B からの直接のご連絡でご案内します。

### ダウンタイムは発生しますか？

移行の間、従来の Model Registry と新しい W&B Registry への書き込み操作は、約1時間停止します。それ以外の W&B サービスは通常通りご利用いただけます。

### コードは壊れますか？

いいえ。従来の Model Registry のパスや Python SDK コールはすべて自動的に新しい Registry にリダイレクトされます。

### データが削除されることはありますか？

ありません。データは新しい W&B Registry にコピーされ、従来の Model Registry は読み取り専用となり、後に非表示となります。データが削除されることはありません。

### 古い SDK を使っている場合は？

リダイレクト自体は機能しますが、警告表示はありません。より良い体験のために最新版 W&B SDK の利用をおすすめします。

### 移行後のRegistryの名前変更やメンバー管理はできますか？

はい、移行後のRegistryでは名前変更やメンバーの追加・削除等の操作が可能です。これらのRegistryはカスタムRegistryとして扱われ、移行後もリダイレクト機能は継続されます。

## ご質問は？

サポートや移行についてのご相談は [support@wandb.com](mailto:support@wandb.com) までご連絡ください。W&B は新しい W&B Registry へのスムーズな移行を全力でサポートします。