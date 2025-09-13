---
title: なぜ私の個人アカウントでは W&B Inference を利用できないのですか？
menu:
  support:
    identifier: ja-support-kb-articles-inference_personal_account_limitation
support:
- 推論
toc_hide: true
type: docs
url: /support/:filename
---

個人アカウントは W&B Inference をサポートしていません。次の 429 エラーが表示されます: "W&B Inference isn't available for personal accounts. Please switch to a non-personal account to access W&B Inference."

## 背景

Personal Entities は 2024 年 5 月に非推奨になりました。これは、Personal Entities をまだ使用しているレガシーアカウントのみに影響します。

## W&B Inference への アクセス 方法

### Team を作成

1. W&B アカウントにログインする
2. 右上のプロフィールアイコンをクリック
3. "Create new team" を選択
4. Team 名を選ぶ
5. W&B Inference のリクエストにこの Team を使用する

### コードを更新

Personal Entity から Team へ変更:

**変更前（動作しません）:**
```python
project="your-username/project-name"  # 個人の Entity
```

**変更後（動作します）:**
```python
project="your-team/project-name"  # Team の Entity
```

## Teams を使用するメリット

- W&B Inference への アクセス
- より強力なコラボレーション機能
- 共有の Projects とリソース
- Team ベースの課金と利用状況の追跡

## お困りですか？

Team の作成や個人アカウントからの切り替えに問題がある場合は、W&B サポートまでお問い合わせください。