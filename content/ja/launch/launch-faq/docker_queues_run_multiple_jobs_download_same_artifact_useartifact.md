---
title: When multiple jobs in a Docker queue download the same artifact, is any caching
  used, or is it re-downloaded every run?
menu:
  launch:
    identifier: ja-launch-launch-faq-docker_queues_run_multiple_jobs_download_same_artifact_useartifact
    parent: launch-faq
---

キャッシュは存在しません。各 Launch ジョブは独立して動作します。キューまたは エージェント を設定して、キューの設定で Docker の 引数 を使用して共有キャッシュをマウントします。

さらに、特定の ユースケース では、W&B Artifacts キャッシュを永続ボリュームとしてマウントします。
