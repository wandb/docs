---
title: Python でのみ動作しますか？
menu:
  support:
    identifier: ja-support-kb-articles-work_python
support:
- "{< tabpane >} {% tab header=\"English\" %}  \n\n`wandb.init()` automatically looks\
  \ for two environment variables: `WANDB_API_KEY` and `WANDB_BASE_URL`. These environment\
  \ variables define the API key and the URL for the W&B server.  \n\nIf you have\
  \ multiple W&B accounts, you can use different API keys by setting the `WANDB_API_KEY`\
  \ environment variable.  \n\nAlternatively, you can use the `login()` function from\
  \ the `wandb` module to authenticate your client. You'll be prompted to enter your\
  \ API key the first time you run `wandb.login()`.  \n\n```python\nimport wandb\n\
  \n# 初めてアカウントにログインすると、APIキーの入力が求められます。\nwandb.login()\n```  \n\n{% /tab %} {% tab\
  \ header=\"Japanese\" %}  \n\n`wandb.init()` は自動的に2つの環境変数を探します: `WANDB_API_KEY`\
  \ と `WANDB_BASE_URL`。これらの環境変数は、APIキーと W&B サーバーのURLを定義します。\n\n複数の W&B アカウントを持っている場合は、異なるAPIキーを使用するために\
  \ `WANDB_API_KEY` 環境変数を設定できます。\n\nあるいは、`wandb` モジュールの `login()` 関数を使用してクライアントを認証することもできます。初めて\
  \ `wandb.login()` を実行すると、APIキーの入力が求められます。\n\n```python\nimport wandb\n\n# 初めてアカウントにログインすると、APIキーの入力が求められます。\n\
  wandb.login()\n```\n\n{% /tab %} {< /tabpane >}"
toc_hide: true
type: docs
url: /ja/support/:filename
---

ライブラリは Python 2.7 以降、および Python 3.6 以降をサポートしています。アーキテクチャーは他のプログラミング言語とのインテグレーションを容易にします。他の言語をモニタリングするには、[contact@wandb.com](mailto:contact@wandb.com) にお問い合わせください。