---
title: What requirements does the accelerator base image have?
menu:
  launch:
    identifier: ja-launch-launch-faq-requirements_accelerator_base_image
    parent: launch-faq
---

アクセラレーターを利用するジョブの場合、必要なアクセラレーターコンポーネントを含むベースイメージを指定してください。アクセラレーターイメージには、以下の要件を満たすようにしてください。

- Debian との互換性 ( Launch の Dockerfile は、apt-get を使用して Python をインストールします)
- サポートされている CPU および GPU ハードウェア命令セット (目的の GPU との CUDA バージョンの互換性を確認してください)
- 提供されるアクセラレーターバージョンと機械学習アルゴリズムのパッケージとの互換性
- ハードウェア互換性のため追加の手順が必要なパッケージのインストール