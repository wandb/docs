---
title: How does W&B Launch build images?
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_build_images
    parent: launch-faq
---

イメージを構築する手順は、ジョブのソースと、リソース設定で指定されたアクセラレータのベースイメージによって異なります。

{{% alert %}}
キューを設定したり、ジョブを送信する際は、キューまたはジョブリソース設定にベースアクセラレータイメージを含めてください。
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

構築プロセスには、ジョブタイプと指定されたアクセラレータのベースイメージに基づいて、次のアクションが含まれます。

| | aptを使用してPythonをインストール | Pythonパッケージをインストール | ユーザーとワークディレクトリを作成 | コードをイメージにコピー | エントリーポイントを設定 | |
