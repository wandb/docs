---
title: なぜローカルでトレーニングが正常に動作しているのに、W&B では run が「crashed」と表示されるのですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- クラッシュやフリーズする run
---

これは接続の問題を示しています。サーバーがインターネット アクセスを失い、データの同期が W&B へ停止した場合、システムは短い再試行期間の後に run をクラッシュとしてマークします。