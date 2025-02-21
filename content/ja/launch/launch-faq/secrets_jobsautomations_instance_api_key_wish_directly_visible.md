---
title: Can you specify secrets for jobs/automations? For instance, an API key which
  you do not wish to be directly visible to users?
menu:
  launch:
    identifier: ja-launch-launch-faq-secrets_jobsautomations_instance_api_key_wish_directly_visible
    parent: launch-faq
---

はい。次のステップに従ってください：

1. run 用に指定されたネームスペースで Kubernetes シークレットを作成します。使用するコマンド：  
   `kubectl create secret -n <namespace> generic <secret_name> <secret_value>`

2. シークレットを作成した後、run が開始する際にシークレットを注入するようにキューを設定します。シークレットを表示できるのはクラスター管理者のみで、エンドユーザーはそれを見ることができません。