---
title: Manage runs
displayed_sidebar: default
---

### 팀으로 run 이동하기

프로젝트 페이지에서:

1. 테이블 탭을 클릭하여 run 테이블을 확장합니다.
2. 체크박스를 클릭하여 모든 run을 선택합니다.
3. **Move**를 클릭합니다: 대상 project는 개인 계정이나 팀 계정 (회원인 경우)에 있을 수 있습니다.

![](/images/app_ui/demo_move_runs.gif)

### 새로운 run을 팀에 전송하기

스크립트에서 entity를 팀으로 설정하세요. "Entity"는 사용자 이름이나 팀 이름을 의미합니다. run을 전송하기 전에 웹 앱에서 entity(개인 계정 또는 팀 계정)를 만드세요.

```python
wandb.init(entity="example-team")
```

팀에 가입하면 **기본 entity**가 업데이트됩니다. 이는 [설정 페이지](https://app.wandb.ai/settings)를 확인하면, 새 프로젝트를 생성할 기본 위치가 방금 가입한 팀으로 표시됨을 의미합니다. 다음은 해당 [설정 페이지](https://app.wandb.ai/settings) 섹션의 예시입니다:

![](/images/app_ui/send_new_runs_to_team.png)