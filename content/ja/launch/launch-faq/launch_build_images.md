---
title: W&B Launch はどのようにイメージをビルドしますか？
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_build_images
    parent: launch-faq
---

イメージのビルド手順は、ジョブソースおよびリソース設定で指定されたアクセラレーターベースイメージによって異なります。

{{% alert %}}
キューの設定やジョブの提出時には、リソース設定内にベースアクセラレーターイメージを含めてください:
```json
{
    "builder": {
        "accelerator": {
            "base_image": "image-name"
        }
    }
}
```
{{% /alert %}}

ビルドプロセスでは、ジョブタイプと指定されたアクセラレーターベースイメージに基づいて、次のアクションが行われます。

| | apt を使って Python をインストール | Python パッケージをインストール | ユーザーと作業ディレクトリを作成 | コードをイメージにコピー | エントリポイントを設定 | |