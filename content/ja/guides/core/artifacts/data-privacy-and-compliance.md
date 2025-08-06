---
title: Artifact データのプライバシーとコンプライアンス
description: W&B ファイルのデフォルト保存場所について学びましょう。センシティブな情報の保存方法や管理方法も紹介します。
menu:
  default:
    identifier: ja-guides-core-artifacts-data-privacy-and-compliance
    parent: artifacts
---

ファイルは Artifacts をログすると、W&B が管理する Google Cloud バケットにアップロードされます。バケットの内容は、保存時と転送時の両方で暗号化されています。Artifact ファイルは、対応する Project へのアクセス権を持つユーザーのみが閲覧できます。

{{< img src="/images/artifacts/data_and_privacy_compliance_1.png" alt="GCS W&B クライアントサーバーダイアグラム" >}}

Artifact のバージョンを削除すると、当社のデータベースでソフト削除としてマークされ、ストレージコストから除外されます。Artifact 全体を削除すると、完全削除のキューに入り、その中身もすべて W&B バケットから削除されます。ファイル削除に関し特別なニーズがある場合は、[カスタマーサポート](mailto:support@wandb.com)までご連絡ください。

マルチテナント環境に置けない機密性の高い Dataset に対しては、クラウドバケットと接続したプライベートな W&B サーバーや、_reference artifacts_ を使うことができます。Reference artifacts は、ファイルの内容を W&B に送信せず、プライベートバケットへの参照のみを追跡します。Reference artifacts は、あなたのバケットやサーバー上のファイルへのリンクを保持します。つまり、W&B が保持するのはファイルに関するメタデータのみで、本体ファイル自体は保存しません。

{{< img src="/images/artifacts/data_and_privacy_compliance_2.png" alt="W&B クライアントサーバークラウドダイアグラム" >}}

Reference artifact は、通常の artifact を作成するのとほぼ同じ方法で作成できます。

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("animals", type="dataset")
# ファイル参照を追加
artifact.add_reference("s3://my-bucket/animals")
```

その他の選択肢については、プライベートクラウドやオンプレミスでの導入について [contact@wandb.com](mailto:contact@wandb.com) へお問い合わせください。