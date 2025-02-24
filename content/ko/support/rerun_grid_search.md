---
title: Can I rerun a grid search?
menu:
  support:
    identifier: ko-support-rerun_grid_search
tags:
- sweeps
- hyperparameter
- runs
toc_hide: true
type: docs
---

그리드 탐색이 완료되었지만 충돌로 인해 일부 W&B Runs을 다시 실행해야 하는 경우, 특정 W&B Runs을 삭제하여 다시 실행하십시오. 그런 다음 [스윕 제어 페이지]({{< relref path="/guides/models/sweeps/sweeps-ui.md" lang="ko" >}})에서 **다시 시작** 버튼을 선택하십시오. 새 스윕 ID를 사용하여 새로운 W&B 스윕 에이전트를 시작하십시오.

완료된 W&B Run 파라미터 조합은 다시 실행되지 않습니다.
