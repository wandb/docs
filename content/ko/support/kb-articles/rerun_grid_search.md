---
title: Can I rerun a grid search?
menu:
  support:
    identifier: ko-support-kb-articles-rerun_grid_search
support:
- sweeps
- hyperparameter
- runs
toc_hide: true
type: docs
url: /support/:filename
---

그리드 검색이 완료되었지만 충돌로 인해 일부 W&B Runs을 다시 실행해야 하는 경우, 다시 실행할 특정 W&B Runs을 삭제하세요. 그런 다음 [스윕 제어 페이지]({{< relref path="/guides/models/sweeps/sweeps-ui.md" lang="ko" >}})에서 **다시 시작** 버튼을 선택합니다. 새 Sweep ID를 사용하여 새로운 W&B Sweep 에이전트를 시작하세요.

완료된 W&B Run 파라미터 조합은 다시 실행되지 않습니다.
