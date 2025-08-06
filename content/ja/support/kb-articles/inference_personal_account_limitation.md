---
title: なぜ個人アカウントでは W&B Inference が利用できないのですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 推論
---

Personal アカウントでは W&B Inference をサポートしていません。この 429 エラーが表示されます: "W&B Inference isn't available for personal accounts. Please switch to a non-personal account to access W&B Inference."

## 背景

Personal Entity は 2024 年 5 月に非推奨となりました。これは、まだ Personal Entity を使用しているレガシーアカウントのみに影響します。

## W&B Inference へのアクセス方法

### Team の作成

1. W&B アカウントにログインします
2. 右上のプロフィールアイコンをクリックします
3. 「Create new team」を選択します
4. Team 名を決めます
5. この Team を W&B Inference リクエストに使用します

### コードの更新

Personal Entity から Team へ変更します:

**変更前（動作しません）:**
```python
project="your-username/project-name"  # Personal entity
```

**変更後（動作します）:**
```python
project="your-team/project-name"  # Team entity
```

## Team 利用のメリット

- W&B Inference が利用可能
- より優れたコラボレーション機能
- プロジェクトやリソースの共有
- Team ベースの請求や利用状況の追跡

## お困りですか？

Team の作成や Personal アカウントからの切り替えでお困りの場合は、W&B サポートまでご連絡ください。