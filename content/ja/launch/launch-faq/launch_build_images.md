---
title: How does W&B Launch build images?
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_build_images
    parent: launch-faq
---

イメージを構築する手順は、ジョブのソースと、リソース設定で指定されたアクセラレータのベースイメージによって異なります。

{{% alert %}}
キューを設定するか、ジョブを送信する際に、キューまたはジョブのリソース設定にベースアクセラレータイメージを含めます。
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

構築プロセスには、ジョブタイプと提供されたアクセラレータのベースイメージに基づいて、次のアクションが含まれます。

| | apt を使用して Python をインストール | Python パッケージをインストール | ユーザー と workdir を作成 | コード をイメージにコピー | エントリーポイント を設定 | |
