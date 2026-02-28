---
title: GitHub Actions 변경 사항 테스트
---

<div id="agent-prompt-testing-github-actions-changes-in-wandbdocs">
  # 에이전트 프롬프트: GitHub Actions 변경 사항을 wandb/docs에서 테스트하기
</div>

<div id="requirements">
  ## 요구 사항
</div>

- **W&B 직원 액세스**: 내부 W&B 시스템에 접근할 수 있는 W&B 직원이어야 합니다.
- **GitHub 포크**: 워크플로 변경 사항을 테스트하기 위한 wandb/docs 저장소의 개인 포크가 필요합니다. 이 포크에서 기본 브랜치에 직접 push할 수 있는 권한과 브랜치 보호 규칙을 우회할 수 있는 권한이 있어야 합니다.

<div id="agent-prerequisites">
  ## 에이전트 사전 준비 사항
</div>

시작하기 전에 다음 정보를 수집하세요:

1. **GitHub 사용자 이름** - 먼저 포크 원격 저장소를 확인하려면 `git remote -v`를, 그다음 사용자 이름을 확인하려면 `git config`를 확인하세요. 두 곳 어디에서도 찾을 수 없는 경우에만 사용자에게 물어보세요.
2. **포크 상태** - 기본 브랜치에 푸시할 수 있고 브랜치 보호를 우회할 수 있는 권한이 있는 wandb/docs 포크를 가지고 있는지 확인하세요.
3. **테스트 범위** - 어떤 구체적인 변경 사항을 테스트 중인지(의존성 업그레이드, 기능 변경 등) 물어보세요.

<div id="task-overview">
  ## 작업 개요
</div>

wandb/docs 리포지토리에서 GitHub Actions 워크플로의 변경 사항을 테스트합니다.

<div id="context-and-constraints">
  ## 맥락 및 제약 조건
</div>

<div id="repository-setup">
  ### 리포지토리 설정
</div>

- **메인 리포지토리**: `wandb/docs` (origin)
- **테스트용 포크**: `<username>/docs` (fork remote) - `git remoter -v` 출력만으로 명확하지 않다면, 사용자의 포크 엔드포인트를 물어봐야 합니다.
- **중요**: PR의 GitHub Actions는 항상 PR 브랜치가 아니라 기준 브랜치(main)에서 실행됩니다.
- **Mintlify 배포 제한 사항**: Mintlify 배포와 `link-rot` 체크는 포크가 아닌 메인 wandb/docs 리포지토리에 대해서만 빌드됩니다. 포크에서는, `validate-mdx` Github Action이 포크 PR에서 `mint dev`와 `mint broken-links` 명령의 상태를 확인합니다.

**에이전트 메모**: 다음을 수행해야 합니다:

1. `git remote -v`를 확인해 기존 fork remote가 있는지 확인하고, 있다면 URL에서 username을 추출합니다.
2. remotes에서 username을 찾지 못한 경우 `git config`에서 GitHub username을 확인합니다.
3. 두 위치 어디에서도 찾지 못한 경우에만 사용자에게 GitHub username을 물어봅니다.
4. 테스트에 사용할 수 있는 wandb/docs 포크가 있는지 확인합니다.
5. 포크에 직접 푸시할 수 없다면, 사용자가 그 브랜치에서 푸시할 수 있도록 wandb/docs에 임시 브랜치를 생성합니다.

<div id="testing-requirements">
  ### 테스트 요구 사항
</div>

워크플로 변경 사항을 테스트하려면 다음을 수행해야 합니다.

1. 모든 임시 커밋을 폐기한 상태로 포크의 `main`을 원본 저장소의 `main`과 동기화합니다.
2. 포크의 기능 브랜치뿐 아니라 `main` 브랜치에도 변경 사항을 적용합니다.
3. 워크플로를 트리거하기 위해 내용 변경을 포함한 테스트 PR을 포크의 `main`을 대상으로 생성합니다.

<div id="step-by-step-testing-process">
  ## 단계별 테스트 절차
</div>

<div id="1-initial-setup">
  ### 1. 초기 설정
</div>

```bash
# 기존 원격 저장소 확인
git remote -v

# fork 원격 저장소가 있으면 fork URL에서 사용자 이름 확인
# fork 원격 저장소가 없으면 git config에서 사용자 이름 확인
git config user.name  # 또는 git config github.user

# 원격 저장소나 config에서 찾을 수 없는 경우에만 사용자에게 GitHub 사용자 이름 또는 fork 정보를 요청
# 질문 예시: "테스트에 사용할 fork의 GitHub 사용자 이름이 무엇인가요?"

# fork 원격 저장소가 없으면 추가:
git remote add fork https://github.com/<username>/docs.git  # <username>을 실제 사용자 이름으로 교체
```


<div id="2-sync-fork-and-prepare-test-branch">
  ### 2. 포크 동기화 및 테스트 브랜치 준비
</div>

```bash
# origin에서 최신 내용 가져오기
git fetch origin

# main 체크아웃 후 origin/main으로 하드 리셋하여 깔끔한 동기화 보장
git checkout main
git reset --hard origin/main

# fork에 강제 푸시하여 동기화 (fork의 임시 커밋 제거)
git push fork main --force

# 워크플로우 변경사항용 테스트 브랜치 생성
git checkout -b test-[description]-[date]
```


<div id="3-apply-workflow-changes">
  ### 3. 워크플로 변경 사항 적용
</div>

워크플로 파일을 수정합니다. 의존성을 업그레이드할 때는 다음을 수행하세요.

- `uses:` 구문에서 버전 번호를 업데이트합니다.
- 의존성이 여러 위치에서 사용된다면 두 워크플로 파일을 모두 확인합니다.

**전문가 팁**: 어떤 런북이든 최종 확정하기 전에, 아래와 같은 프롬프트로 AI 에이전트에게 검토를 요청하세요.

> "이 런북을 검토하고 AI 에이전트에게 더 유용해지도록 개선점을 제안해 주세요. 명확성, 완전성, 그리고 모호성 제거에 초점을 맞춰 주세요."

<div id="5-commit-and-push-to-forks-main">
  ### 5. 포크된 main 브랜치에 커밋하고 푸시하기
</div>

```bash
# 모든 변경사항 커밋
git add -A
git commit -m "test: [Description of what you're testing]"

# fork의 main 브랜치에 푸시
git push fork HEAD:main --force-with-lease
```

**포크 접근을 위한 에이전트 지침**:
포크에 직접 푸시할 수 없는 경우:

1. 변경 사항이 포함된 임시 브랜치를 wandb/docs 저장소에 생성합니다.
2. 사용자에게 다음 명령을 전달합니다:
   ```bash
   git fetch origin temp-branch-name
   git push fork origin/temp-branch-name:main --force
   ```
3. 다음 주소에서 PR을 생성하도록 안내합니다: `https://github.com/<username>/docs/compare/main...test-pr-[description]`
4. 테스트가 끝난 후 wandb/docs에서 임시 브랜치를 삭제하는 것을 잊지 마세요.


<div id="6-create-test-pr">
  ### 6. 테스트용 PR 생성
</div>

```bash
# 업데이트된 포크 main에서 새 브랜치 생성
git checkout -b test-pr-[description]

# 워크플로우를 트리거할 소규모 콘텐츠 변경
echo "<!-- Test change for PR preview -->" >> content/en/guides/quickstart.md

# 커밋 및 푸시
git add content/en/guides/quickstart.md
git commit -m "test: PR 미리보기를 트리거하기 위한 콘텐츠 변경 추가"
git push fork test-pr-[description]
```

그런 다음 GitHub UI에서 `<username>:test-pr-[description]` 브랜치에서 `<username>:main` 브랜치로 가는 PR을 생성합니다


<div id="7-monitor-and-verify">
  ### 7. 모니터링 및 검증
</div>

기대되는 동작:

1. GitHub Actions 봇이 "Generating preview links..."라는 초기 코멘트를 생성한다.
2. 워크플로가 오류 없이 완료된다.

다음을 확인:

- ✅ 워크플로가 성공적으로 완료됨
- ✅ 프리뷰 코멘트가 생성되고 업데이트됨
- ✅ 링크가 오버라이드 URL을 사용함
- ✅ 파일 분류가 정상 동작함 (Added/Modified/Deleted/Renamed)
- ❌ Actions 로그에 오류가 있음
- ❌ 보안 경고 또는 노출된 시크릿이 있음

<div id="8-cleanup">
  ### 8. 정리
</div>

테스트를 마친 후에는:

```bash
# 포크의 main을 upstream과 일치하도록 초기화
git checkout main
git fetch origin
git reset --hard origin/main
git push fork main --force

# 포크와 origin에서 테스트 브랜치 삭제
git branch -D test-[description]-[date] test-pr-[description]
```


<div id="common-issues-and-solutions">
  ## 자주 발생하는 문제와 해결 방법
</div>

<div id="issue-permission-denied-when-pushing-to-fork">
  ### 문제: 포크 저장소에 푸시할 때 Permission denied 발생
</div>

- GitHub 토큰이 읽기 전용일 수 있음
- 해결 방법: SSH를 사용하거나 로컬 머신에서 직접 푸시하세요

<div id="issue-workflows-not-triggering">
  ### 문제: 워크플로우가 트리거되지 않음
</div>

- 참고: 워크플로우는 PR 브랜치가 아니라 기본 브랜치(main)에서 실행됩니다.
- 변경 사항이 포크의 기본 브랜치(main)에 있는지 확인하세요.

<div id="issue-changed-files-not-detected">
  ### 문제: 변경된 파일이 감지되지 않음
</div>

- 콘텐츠 변경 사항이 추적 중인 디렉터리(content/, static/, assets/ 등)에 있는지 확인하세요.
- 워크플로 설정의 `files:` 필터를 확인하세요.

<div id="testing-checklist">
  ## 테스트 체크리스트
</div>

- [ ] 사용자에게 GitHub 사용자명과 포크 관련 정보를 요청했는가
- [ ] origin과 포크 두 원격 저장소가 모두 설정되었는가
- [ ] 워크플로우 변경 사항이 두 개의 관련 파일 모두에 적용되었는가
- [ ] 변경 사항이 포크의 main 브랜치에 푸시되었는가(직접 또는 사용자를 통해)
- [ ] 콘텐츠 변경 사항으로 테스트 PR이 생성되었는가
- [ ] 미리보기 댓글이 성공적으로 생성되었는가
- [ ] GitHub Actions 로그에 오류가 없는가
- [ ] 테스트 후 포크의 main 브랜치가 초기 상태로 되돌려졌는가
- [ ] 임시 브랜치가 wandb/docs에서 정리되었는가(생성된 경우)