---
title: Can you group runs by tags?
menu:
  support:
    identifier: ko-support-kb-articles-group_runs_tags
support:
- runs
toc_hide: true
type: docs
url: /support/:filename
---

하나의 run에 여러 개의 태그가 있을 수 있으므로 태그별 그룹화는 지원되지 않습니다. 이러한 run에 대한 [`config`]({{< relref path="/guides/models/track/config.md" lang="ko" >}}) 오브젝트에 값을 추가하고 대신 이 config 값으로 그룹화하세요. 이는 [API]({{< relref path="/guides/models/track/config.md#set-the-configuration-after-your-run-has-finished" lang="ko" >}})를 사용하여 수행할 수 있습니다.
