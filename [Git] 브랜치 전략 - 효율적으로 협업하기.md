Git을 어느 정도 사용해본 개발자라면 한 번쯤 들어봤을 **“Git 브랜치 전략”**.

혼자 작업할 땐 단순한 브랜치 관리로 충분하지만, **팀 단위 협업**에서는 브랜치를 체계적으로 나눠야 **충돌 없이, 안정적이고 유연한 개발**이 가능해집니다.

이번 포스팅에서는 실무에서도 많이 쓰이는 **Git Flow 스타일**의 브랜치 전략을 중심으로, 각 브랜치의 역할과 사용 시나리오를 정리해볼게요.

---

## 📌 목차

1. Git 브랜치 전략이 필요한 이유
2. 주요 브랜치 소개
3. 브랜치 간 흐름과 릴리즈 플로우
4. 실제 협업 예시
5. 전략 적용 팁 & 주의사항
6. 마무리

---

## 1. 왜 브랜치 전략이 필요할까?

협업에서 브랜치를 잘못 관리하면 다음과 같은 문제들이 생깁니다:

- 긴급 수정 사항 반영 중 기능 개발 내용이 섞임
- QA 중 테스트 대상 커밋이 명확하지 않음
- 배포 시점이 제각각이라 혼란 발생

→ 이런 문제를 해결하려면 **역할별 브랜치 전략이 필수**입니다!

---

## 2. 주요 브랜치 소개

### `main` (또는 `master`)

- **프로덕션(실서비스)에서 사용하는 코드**
- 항상 **안정적인 상태**여야 함
- 배포는 `main`에서 진행됨

---

### `develop`

- 기능 개발을 통합하는 **중간 브랜치**
- 여러 `feature` 브랜치가 여기로 병합됨
- QA 또는 스테이징 서버에 반영할 기준

> 일부 팀은 develop 없이 main + feature 구조를 쓰기도 함
> 

---

### `feature/<기능명>`

- 특정 기능 단위 개발용 브랜치
- `develop` 또는 `main`에서 분기
- 기능 개발 완료 후 `develop`으로 병합

```bash
git checkout -b feature/login develop
```

---

### `release/<버전명>`

- 릴리즈 전 최종 검토 및 QA 진행
- 버그 수정, 문서 보완 등만 허용
- 완료되면 `main`과 `develop`에 병합

```bash
git checkout -b release/1.0.0 develop
```

---

### `hotfix/<이슈명>`

- 배포 후 발견된 **긴급 이슈 수정용 브랜치**
- `main`에서 바로 분기하여 수정
- 수정 완료 시 `main`과 `develop`에 병합

```bash
git checkout -b hotfix/typo-fix main
```

---

## 3. 브랜치 간 흐름

```
              +-------------------+
              |    hotfix/*       |
              +---------|---------+
                        v
  develop <--- feature/* |       release/* ---> main
             ------------+--------------------->
```

### 일반적인 흐름 요약:

1. 기능 개발: `feature/*` → `develop`
2. QA 및 릴리즈 준비: `develop` → `release/*`
3. 릴리즈 완료: `release/*` → `main` + `develop`
4. 긴급 패치: `main` → `hotfix/*` → `main` + `develop`

---

## 4. 실제 협업 예시

### 신규 기능 개발 & 릴리즈 시나리오

```bash
# 1. develop에서 기능 브랜치 생성
git checkout develop
git checkout -b feature/payment

# 2. 작업 후 develop에 병합
git checkout develop
git merge feature/payment

# 3. 릴리즈 브랜치 생성
git checkout -b release/1.1.0

# 4. 테스트 완료 후 main에 병합
git checkout main
git merge release/1.1.0

# 5. develop에도 반영
git checkout develop
git merge release/1.1.0

# 6. release 브랜치 삭제
git branch -d release/1.1.0
```

---

### 긴급 수정이 필요할 때

```bash
# 1. main에서 hotfix 분기
git checkout main
git checkout -b hotfix/crash-fix

# 2. 수정 후 main에 병합
git checkout main
git merge hotfix/crash-fix

# 3. develop에도 병합
git checkout develop
git merge hotfix/crash-fix
```

---

## 5. 전략 적용 팁 & 주의사항

### ✅ 팁

- 브랜치 이름은 **일관된 네이밍** 유지 (`feature/login`, `hotfix/null-error`)
- 기능 단위 브랜치는 가급적 **짧은 시간 안에** 병합
- 릴리즈는 반드시 QA 및 테스트 완료 후 배포
- 커밋 정리는 `rebase` 또는 `squash`로 병합 전에 깔끔하게!

---

### ⚠️ 주의사항

- `main`은 항상 **배포 가능한 상태**여야 함
- `hotfix`를 `develop`에 병합하지 않으면 다음 릴리즈에 빠질 수 있음
- 브랜치가 너무 많아지면 관리 어려움 → 주기적으로 정리

---

## 6. 마무리

Git 브랜치 전략을 잘 세워두면:

- 협업이 명확해지고
- 버그 수정/배포 시 혼란이 줄고
- 기능/릴리즈 단위 관리가 체계적으로 가능

특히 팀 규모가 커질수록 브랜치 전략의 중요성은 더욱 커집니다.

---
