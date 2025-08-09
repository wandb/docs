---
title: run 을 태그로 그룹화할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-group_runs_tags
support:
- run
toc_hide: true
type: docs
url: /support/:filename
---

하나의 run 에 여러 개의 태그를 추가할 수 있기 때문에 태그로 그룹화는 지원되지 않습니다. 이러한 run 에는 [`config`]({{< relref path="/guides/models/track/config.md" lang="ko" >}}) 오브젝트에 값을 추가하고, 이 config 값으로 그룹화하세요. 이는 [API]({{< relref path="/guides/models/track/config.md#set-the-configuration-after-your-run-has-finished" lang="ko" >}})를 사용하여 할 수 있습니다.