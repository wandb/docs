---
description: W&Bファイルがデフォルトでどこに保存されるかを学びます。敏感な情報を保存し、保管する方法を探ります。
displayed_sidebar: default
---


# Data Privacy and Compliance

<head>
    <title>Artifact Data Privacy and Compliance</title>
</head>

ファイルは Artifacts をログする際に W&B によって管理された Google Cloud バケットにアップロードされます。バケットの内容は静止時と転送時の両方で暗号化されます。Artifact ファイルは、対応するプロジェクトにアクセス権を持つユーザーのみに表示されます。

![GCS W&B Client Server diagram](/images/artifacts/data_and_privacy_compliance_1.png)

アーティファクトのバージョンを削除すると、それはデータベースでソフト削除としてマークされ、ストレージコストからも除外されます。アーティファクト全体を削除する場合、それは永久削除のためにキューに登録され、その内容はすべて W&B バケットから削除されます。ファイル削除に関して具体的な要望がある場合は、[カスタマーサポート](mailto:support@wandb.com)までご連絡ください。

マルチテナント環境に存在できない機密データセットについては、クラウドバケットに接続されたプライベート W&B サーバーまたは _reference artifacts_ を使用できます。Reference artifacts は、ファイルコンテンツを W&B に送信せずにプライベートバケットへの参照を追跡します。Reference artifacts は、バケットやサーバー上のファイルへのリンクを保持します。言い換えると、W&B はファイル自体ではなく、ファイルに関連するメタデータのみを追跡します。

![W&B Client Server Cloud diagram](/images/artifacts/data_and_privacy_compliance_2.png)

非リファレンスアーティファクトを作成する場合と同様に、リファレンスアーティファクトを作成します:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("animals", type="dataset")
artifact.add_reference("s3://my-bucket/animals")
```

代替案については、プライベートクラウドやオンプレミスのインストールについて話し合うために [contact@wandb.com](mailto:contact@wandb.com) までご連絡ください。