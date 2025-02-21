---
title: What requirements does the accelerator base image have?
menu:
  launch:
    identifier: ja-launch-launch-faq-requirements_accelerator_base_image
    parent: launch-faq
---

アクセラレータを利用するジョブの場合は、必要なアクセラレータコンポーネントを含むベースイメージを指定してください。アクセラレータイメージには、以下の要件を満たす必要があります。

- Debian との互換性 ( Launch の Dockerfile は apt-get を使用して Python をインストールします)
- サポートされている CPU および GPU ハードウェアの命令セット (意図する GPU との CUDA バージョンの互換性を確認してください)
- 提供されるアクセラレータのバージョンと、機械学習アルゴリズムのパッケージとの互換性
- ハードウェアの互換性のため、追加の手順が必要となるパッケージのインストール