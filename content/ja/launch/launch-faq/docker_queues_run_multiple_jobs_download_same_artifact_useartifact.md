---
title: Docker のキュー内で複数のジョブが同じ Artifact をダウンロードする場合、キャッシュは使われますか？それとも各 run ごとに再ダウンロードされますか？
menu:
  launch:
    identifier: ja-launch-launch-faq-docker_queues_run_multiple_jobs_download_same_artifact_useartifact
    parent: launch-faq
---

キャッシュはありません。各 Launch ジョブは独立して動作します。キューの設定で Docker の引数を用いて、キューまたはエージェントが共有キャッシュをマウントするように設定してください。

さらに、特定のユースケースに応じて、W&B Artifacts のキャッシュを永続ボリュームとしてマウントしてください。