---
title: 그리드 검색을 다시 실행할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-rerun_grid_search
support:
- 스윕
- 하이퍼파라미터
- run
toc_hide: true
type: docs
url: /support/:filename
---

그리드 검색이 완료되었지만 일부 W&B Run에서 오류로 인해 재실행이 필요한 경우, 해당 W&B Run을 삭제한 후 다시 실행하세요. 그런 다음, [sweep control page]({{< relref path="/guides/models/sweeps/sweeps-ui.md" lang="ko" >}})에서 **Resume** 버튼을 선택하세요. 새로운 Sweep ID로 새로운 W&B Sweep 에이전트를 시작하면 됩니다.

완전히 종료된 W&B Run 파라미터 조합은 다시 실행되지 않습니다.