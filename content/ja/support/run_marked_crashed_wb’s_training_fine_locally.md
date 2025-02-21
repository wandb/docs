---
title: Why is a run marked crashed in W&B when it’s training fine locally?
menu:
  support:
    identifier: ja-support-run_marked_crashed_wb’s_training_fine_locally
tags:
- crashing and hanging runs
toc_hide: true
type: docs
---

これは接続の問題を示しています。 サーバー がインターネット アクセス を失い、 データ が W&B に同期されなくなると、システムは短いリトライ期間の後、 run をクラッシュしたとマークします。
