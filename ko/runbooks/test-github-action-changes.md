# Agent prompt: wandb/docs의 GitHub Actions 변경 사항 테스트하기

## 요구 사항
- **W&B 직원 엑세스**: 내부 W&B 시스템에 대한 엑세스 권한이 있는 W&B 직원이어야 합니다.
- **GitHub fork**: 워크플로우 변경 사항을 테스트하기 위한 wandb/docs의 개인 fork가 필요합니다. fork에서는 기본 브랜치에 푸시하고 브랜치 보호 규칙을 우회할 수 있는 권한이 필요합니다.

## Agent 사전 준비 사항
시작하기 전에 다음 정보를 수집하세요:
1. **GitHub 사용자 이름** - 먼저 `git remote -v`에서 fork 리모트를 확인한 다음, `git config`에서 사용자 이름을 확인하세요. 두 곳 모두에서 찾을 수 없는 경우에만 사용자에게 요청하세요.
2. **Fork 상태** - wandb/docs의 fork가 있는지, 기본 브랜치에 푸시하고 브랜치 보호를 우회할 수 있는 권한이 있는지 확인하세요.
3. **테스트 범위** - 어떤 구체적인 변경 사항(의존성 업그레이드, 기능 변경 등)을 테스트하는지 확인하세요.

## 작업 개요
wandb/docs 저장소의 GitHub Actions 워크플로우 변경 사항을 테스트합니다.


## 컨텍스트 및 제약 사항

### 저장소 설정
- **메인 저장소**: `wandb/docs` (origin)
- **테스트용 fork**: `<username>/docs` (fork 리모트) - `git remote -v`에서 명확하지 않은 경우, 사용자에게 fork 엔드포인트를 요청하세요.
- **중요**: PR의 GitHub Actions는 항상 PR 브랜치가 아닌 베이스 브랜치(main)에서 실행됩니다.
- **Mintlify 배포 제한**: Mintlify 배포 및 `link-rot` 체크는 오직 메인 wandb/docs 저장소에 대해서만 빌드되며, fork에서는 빌드되지 않습니다. fork의 PR에서는 `validate-mdx` GitHub Action이 `mint dev` 및 `mint broken-links` 코맨드의 상태를 체크합니다.

**Agent 참고**: 다음 사항을 수행해야 합니다:
1. `git remote -v`에서 기존 fork 리모트를 확인하고 URL이 있다면 사용자 이름을 추출하세요.
2. 리모트에서 사용자 이름을 찾을 수 없는 경우, `git config`에서 GitHub 사용자 이름을 확인하세요.
3. 두 곳 모두에서 찾을 수 없는 경우에만 사용자에게 GitHub 사용자 이름을 요청하세요.
4. 테스트에 사용할 수 있는 wandb/docs의 fork가 있는지 확인하세요.
5. fork에 직접 푸시할 수 없는 경우, 사용자가 푸시할 수 있도록 wandb/docs에 임시 브랜치를 생성하세요.

### 테스트 요구 사항
워크플로우 변경 사항을 테스트하려면 다음을 수행해야 합니다:
1. fork의 `main`을 메인 저장소의 `main`과 동기화하여 모든 임시 커밋을 삭제합니다.
2. 변경 사항을 기능 브랜치가 아닌 fork의 main 브랜치에 적용합니다.
3. 워크플로우를 트리거하기 위해 콘텐츠 변경 사항을 포함하여 fork의 `main`을 대상으로 테스트 PR을 생성합니다.

## 단계별 테스트 프로세스

### 1. 초기 설정
```bash
# 기존 리모트 확인
git remote -v

# fork 리모트가 있으면 fork URL에서 사용자 이름을 기록합니다.
# fork 리모트가 없으면 git config에서 사용자 이름을 확인합니다.
git config user.name  # 또는 git config github.user

# 리모트나 설정에서 찾을 수 없는 경우에만 사용자에게 GitHub 사용자 이름이나 fork 정보를 요청합니다.
# 질문 예시: "테스트에 사용할 fork의 GitHub 사용자 이름이 무엇인가요?"

# fork 리모트가 없는 경우 추가합니다:
git remote add fork https://github.com/<username>/docs.git  # <username>을 실제 이름으로 바꿉니다.
```

### 2. Fork 동기화 및 테스트 브랜치 준비
```bash
# origin에서 최신 정보 가져오기
git fetch origin

# main을 체크아웃하고 origin/main으로 하드 리셋하여 깨끗하게 동기화합니다.
git checkout main
git reset --hard origin/main

# fork에 강제 푸시하여 동기화합니다 (fork의 모든 임시 커밋을 삭제함).
git push fork main --force

# 워크플로우 변경을 위한 테스트 브랜치를 생성합니다.
git checkout -b test-[설명]-[날짜]
```

### 3. 워크플로우 변경 사항 적용
워크플로우 파일을 수정합니다. 의존성 업그레이드의 경우:
- `uses:` 구문의 버전 번호를 업데이트합니다.
- 의존성이 여러 곳에서 사용되는 경우 두 워크플로우 파일을 모두 확인하세요.

**전문가 팁**: runbook을 확정하기 전에 AI 에이전트에게 다음과 같은 프롬프트로 검토를 요청하세요:
> "이 runbook을 검토하고 AI 에이전트에게 더 유용하도록 개선 사항을 제안해 주세요. 명확성, 완전성에 집중하고 모호함을 제거해 주세요."

### 5. 커밋 및 fork의 main으로 푸시
```bash
# 모든 변경 사항 커밋
git add -A
git commit -m "test: [테스트 내용 설명]"

# fork의 main 브랜치로 푸시
git push fork HEAD:main --force-with-lease
```

**fork 엑세스에 대한 Agent 안내**:
fork에 직접 푸시할 수 없는 경우:
1. 변경 사항이 포함된 임시 브랜치를 wandb/docs에 생성합니다.
2. 사용자에게 다음 코맨드를 제공합니다:
   ```bash
   git fetch origin temp-branch-name
   git push fork origin/temp-branch-name:main --force
   ```
3. 다음 위치에서 PR을 생성하도록 안내합니다: `https://github.com/<username>/docs/compare/main...test-pr-[설명]`
4. 테스트 완료 후 wandb/docs에서 임시 브랜치를 삭제하는 것을 잊지 마세요.


### 6. 테스트 PR 생성
```bash
# 업데이트된 fork main에서 새 브랜치 생성
git checkout -b test-pr-[설명]

# 워크플로우를 트리거하기 위해 작은 콘텐츠 변경을 수행합니다.
echo "
" >> content/en/guides/quickstart.md

# 커밋 및 푸시
git add content/en/guides/quickstart.md
git commit -m "test: PR 미리보기를 트리거하기 위한 콘텐츠 변경 추가"
git push fork test-pr-[설명]
```

그 다음 GitHub UI를 통해 `<username>:test-pr-[설명]`에서 `<username>:main`으로 PR을 생성합니다.

### 7. 모니터링 및 확인

예상 행동:
1. GitHub Actions 봇이 "Generating preview links..."라는 내용으로 초기 댓글을 생성합니다.
2. 워크플로우가 오류 없이 완료되어야 합니다.

확인 사항:
- ✅ 워크플로우가 성공적으로 완료됨
- ✅ 미리보기 댓글이 생성되고 업데이트됨
- ✅ 링크가 override URL을 사용함
- ✅ 파일 분류가 작동함 (Added/Modified/Deleted/Renamed)
- ❌ Actions 로그에 오류가 없음
- ❌ 보안 경고나 노출된 비밀 정보가 없음

### 8. 정리
테스트 완료 후:
```bash
# 업스트림과 일치하도록 fork의 main 리셋
git checkout main
git fetch origin
git reset --hard origin/main
git push fork main --force

# fork와 origin에서 테스트 브랜치 삭제
git branch -D test-[설명]-[날짜] test-pr-[설명]
```

## 일반적인 문제 및 해결 방법

### 문제: fork에 푸시할 때 Permission denied 발생
- GitHub 토큰이 읽기 전용일 수 있습니다.
- 해결 방법: SSH를 사용하거나 로컬 머신에서 수동으로 푸시하세요.

### 문제: 워크플로우가 트리거되지 않음
- 참고: 워크플로우는 PR 브랜치가 아닌 베이스 브랜치(main)에서 실행됩니다.
- 변경 사항이 fork의 main 브랜치에 있는지 확인하세요.

### 문제: 변경된 파일이 감지되지 않음
- 콘텐츠 변경 사항이 추적되는 디렉토리(content/, static/, assets/ 등)에 있는지 확인하세요.
- 워크플로우 설정의 `files:` 필터를 확인하세요.

## 테스트 체크리스트

- [ ] 사용자에게 GitHub 사용자 이름 및 fork 정보를 요청함
- [ ] 두 리모트(origin 및 fork)가 모두 설정됨
- [ ] 두 관련 파일에 워크플로우 변경 사항이 적용됨
- [ ] 변경 사항이 fork의 main 브랜치에 푸시됨 (직접 또는 사용자를 통해)
- [ ] 콘텐츠 변경 사항이 포함된 테스트 PR이 생성됨
- [ ] 미리보기 댓글이 성공적으로 생성됨
- [ ] GitHub Actions 로그에 오류가 없음
- [ ] 테스트 후 fork의 main 브랜치가 리셋됨
- [ ] wandb/docs에서 생성된 임시 브랜치가 정리됨