---
title: なぜ私の個人アカウントで W&B Inference が利用できないのですか？
menu:
  support:
    identifier: ja-support-kb-articles-inference_personal_account_limitation
support:
- 推論
toc_hide: true
type: docs
url: /support/:filename
---

個人アカウントでは W&B Inference は利用できません。この 429 エラーが表示されます:"W&B Inference isn't available for personal accounts. Please switch to a non-personal account to access W&B Inference."

## 背景

Personal Entities は 2024 年 5 月に廃止されました。これは、まだ Personal Entities を利用しているレガシーアカウントにのみ影響します。

## W&B Inference へのアクセス方法

### Team の作成

1. W&B アカウントにログインします
2. 右上のプロフィールアイコンをクリックします
3. 「Create new team」を選択します
4. チーム名を決めます
5. この Team を W&B Inference のリクエストに利用します

### コードの更新

Personal Entity から Team へ切り替えるには以下のように変更します:

**変更前（動作しません）:**
```python
project="your-username/project-name"  # Personal entity
```

**変更後（動作します）:**
```python
project="your-team/project-name"  # Team entity
```

## Teams 利用のメリット

- W&B Inference へのアクセス
- 優れたコラボレーション機能
- 複数人でのプロジェクト・リソース共有
- チーム単位の請求・使用状況トラッキング

## サポートが必要な場合

Team の作成や個人アカウントからの切り替えでお困りの場合は、W&B サポートまでご連絡ください。