---
displayed_sidebar: default
---

# 런 관리

### 런을 팀으로 이동

프로젝트 페이지에서:

1. 런 테이블을 확장하려면 테이블 탭을 클릭하세요.
2. 모든 런을 선택하려면 체크박스를 클릭하세요.
3. **이동**을 클릭하세요: 목적지 프로젝트는 개인 계정이나 소속된 팀 중 하나일 수 있습니다.

![](/images/app_ui/demo_move_runs.gif)

### 새 런을 팀으로 보내기

스크립트에서, 엔티티를 팀으로 설정하세요. "엔티티"는 단순히 사용자 이름이나 팀 이름을 의미합니다. 런을 보내기 전에 웹 앱에서 엔티티(개인 계정 또는 팀 계정)를 생성하세요.

```python
wandb.init(entity="example-team")
```

**기본 엔티티**는 팀에 가입할 때 업데이트됩니다. 이는 [설정 페이지](https://app.wandb.ai/settings)에서 새 프로젝트를 생성할 기본 위치가 방금 가입한 팀으로 변경되었음을 의미합니다. 다음은 해당 [설정 페이지](https://app.wandb.ai/settings) 섹션이 어떻게 보이는지에 대한 예시입니다:

![](/images/app_ui/send_new_runs_to_team.png)