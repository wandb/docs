---
title: Artifact data privacy and compliance
description: デフォルトで W&B ファイルが保存される場所を学びましょう。保存方法や機密情報の保管について探ってみましょう。
menu:
  default:
    identifier: ja-guides-core-artifacts-data-privacy-and-compliance
    parent: artifacts
---

ファイルは Artifacts をログすると、 W&B 管理の Google Cloud バケットにアップロードされます。バケットの内容は、保存時と転送時の両方で暗号化されます。アーティファクトファイルは、対応するプロジェクトへのアクセス権を持つユーザーのみが見ることができます。

{{< img src="/images/artifacts/data_and_privacy_compliance_1.png" alt="GCS W&B Client Server diagram" >}}

アーティファクトのバージョンを削除すると、それはデータベース上でソフト削除にマークされ、ストレージコストから外されます。アーティファクト全体を削除すると、それは完全に削除されるようにキューに入れられ、すべての内容が W&B バケットから削除されます。ファイル削除について特定のニーズがある場合は、[Customer Support](mailto:support@wandb.com) までお問い合わせください。

マルチテナント環境に置くことができない機密データセットの場合は、クラウドバケットに接続されたプライベートな W&B サーバーまたは _reference artifacts_ を使用できます。Reference artifacts は、ファイルの内容を W&B に送信することなく、プライベートバケットへの参照を追跡します。Reference artifacts は、あなたのバケットまたはサーバー上のファイルへのリンクを維持します。つまり、W&B はファイルそれ自体ではなく、ファイルに関連するメタデータのみを記録します。

{{< img src="/images/artifacts/data_and_privacy_compliance_2.png" alt="W&B Client Server Cloud diagram" >}}

Reference artifact を作成するのは、非 reference artifact を作成するのと似ています。

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("animals", type="dataset")
artifact.add_reference("s3://my-bucket/animals")
```

代替案については、[contact@wandb.com](mailto:contact@wandb.com) までお問い合わせいただき、プライベートクラウドやオンプレミスインストールについてご相談ください。