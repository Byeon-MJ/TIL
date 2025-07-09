Git에서 협업을 하다 보면 **로컬 브랜치**뿐 아니라 **원격(remote) 브랜치**를 잘 관리하는 것이 중요합니다.

혼자만 쓰는 브랜치가 아니라, **여러 개발자들과 공유되는 브랜치**이기 때문에 잘못 관리하면 혼란이 생길 수 있죠.

이번 포스팅에서는 **원격 브랜치의 확인, 동기화, 삭제, 추적 브랜치 관리**까지

Git에서 자주 쓰이는 원격 브랜치 관리 명령어들을 모두 정리해드립니다!

---

## 📌 목차

1. 원격 브랜치란?
2. 원격 브랜치 목록 확인하기
3. 원격 브랜치 가져오기(fetch/pull)
4. 원격 브랜치로 push 하기
5. 원격 브랜치 삭제하기
6. 추적 브랜치란?
7. 실전 팁 & 주의사항
8. 마무리

---

## 1. 원격 브랜치란?

원격 브랜치는 Git 서버(GitHub, GitLab 등)에 존재하는 브랜치로, **다른 개발자와 공유되는 브랜치**입니다.

> origin/브랜치명 형태로 로컬에 복사본이 존재하며, 직접 수정할 수 없습니다.
> 
> 
> 변경하려면 로컬 브랜치에서 push하거나 fetch/pull 해야 합니다.
> 

---

## 2. 원격 브랜치 목록 확인하기

```bash
git branch -r
```

- `r`: 원격(remote)의 브랜치 목록만 표시

예시 출력:

```
origin/HEAD -> origin/main
origin/main
origin/feature/login
origin/release/1.0.0
```

### 모든 브랜치(로컬 + 원격) 확인

```bash
git branch -a
```

---

## 3. 원격 브랜치 가져오기

### `fetch`: 변경사항만 가져오기 (병합 X)

```bash
git fetch origin
```

- 원격 저장소의 최신 브랜치 정보를 **로컬에 반영** (HEAD는 그대로)

---

### `pull`: 변경사항을 현재 브랜치에 병합

```bash
git pull origin main
```

- fetch + merge 과정을 자동으로 수행

---

## 4. 로컬 브랜치를 원격 브랜치로 push 하기

### 브랜치 생성과 동시에 push

```bash
git push origin feature/new-ui
```

- `feature/new-ui`라는 원격 브랜치가 생성됨
- 로컬 브랜치와 원격 브랜치 연결됨

---

### 브랜치 연결 확인

```bash
git branch -vv
```

- 추적 중인 브랜치 정보 확인 가능

---

## 5. 원격 브랜치 삭제하기

```bash
git push origin --delete feature/old-test
```

- `feature/old-test` 브랜치가 원격에서 삭제됨

---

### 로컬에서 사라진 원격 브랜치 정보 정리

```bash
# 현재 remote에 대해서 정리, 일반적으로는 origin
git fetch --prune

# 모든 remote에서 정리
git fetch --all --prune
```

- 더 이상 존재하지 않는 원격 브랜치들을 로컬에서 정리해줌

---

## 6. 추적 브랜치란?

추적 브랜치는 **로컬 브랜치와 원격 브랜치를 연결**해주는 역할을 합니다.

예:

```bash
git checkout -b develop origin/develop
```

또는:

```bash
git checkout --track origin/feature/search
```

→ 이후 `git pull`이나 `git push`를 브랜치명 없이도 사용할 수 있음

---

## 7. 실전 팁 & 주의사항

### ✅ 추천 전략

- 브랜치 삭제 전 꼭 팀원과 공유
- 오래된 브랜치는 주기적으로 `fetch --prune`으로 정리
- `push --delete`는 **권한자만** 사용하는 걸 추천
- 기능 작업 시엔 `origin/develop`을 기준으로 새로운 브랜치 생성
- 브랜치 생성 후에는 바로 `push`해서 공유하는 습관 들이기

---

## 흐름 요약 다이어그램

```
로컬 작업 → git push origin <branch>
                    ↓
 원격 저장소 (origin/<branch>)
                    ↓
다른 개발자 → git fetch / git pull
                    ↓
    작업 → 브랜치 정리 / 병합 / 삭제
```

---

## 8. 마무리

Git에서 원격 브랜치 관리까지 익히면 이제 진짜 실무에서도 무리 없이 협업할 수 있는 단계입니다.

브랜치 삭제, push, pull, fetch만 잘 익혀도 **효율적인 협업과 깔끔한 브랜치 운영**이 가능해집니다.

---
