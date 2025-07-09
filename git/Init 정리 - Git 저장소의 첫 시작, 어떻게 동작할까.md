Git을 배우는 사람들이 처음 마주치는 명령어는 단연 `git init`입니다.

하지만 단순히 “Git 시작 명령” 정도로만 알고 있다면, 실제 내부에서 **무슨 일이 일어나는지**, 그리고 **어디까지 자동으로 준비되는지** 정확히 이해하기 어렵습니다.

이번 포스팅에서는 `git init` 명령어의 의미부터, 사용법, 내부 구조, 주의사항까지

Git 저장소의 시작점에 대해 확실히 정리해드립니다!

---

## 📌 목차

1. `git init`이란?
2. `git init`의 동작 원리
3. `git init` 명령 실행 예시
4. `.git/` 내부 디렉토리 구조
5. 기본 브랜치 이름을 main으로 바꾸기
6. 원격 저장소 등록 및 push
7. 실전 초기화 템플릿
8. 주의사항 & 실전 팁
9. 마무리

---

## 1. `git init`이란?

`git init`은 **현재 디렉토리를 Git 저장소로 초기화**해주는 명령어입니다.

> 즉, **"이 폴더 안의 모든 변경 사항을 Git으로 추적할 준비가 되었다"**고 선언하는 명령어입니다.
> 

---

## 2. `git init`의 동작 원리

명령어를 실행하면 다음과 같은 일이 발생합니다:

- `.git/` 디렉토리 생성
- Git 저장소에 필요한 내부 파일 및 구조 생성
- 아직 커밋은 없음 (`HEAD`는 비어 있음)
- 로컬 저장소만 존재 (원격 저장소는 연결되지 않음)

> ⚠️ 단지 Git 저장소가 초기화될 뿐, 아직 트래킹 중인 파일이나 커밋은 없음에 주의!
> 

---

## 3. `git init` 명령 실행 예시

### 기본 사용법

```bash
git init
```

- 현재 디렉토리에서 실행하면 해당 디렉토리가 Git 저장소로 초기화됩니다.

### 새 폴더 생성과 동시에 저장소 초기화

```bash
git init my-project
cd my-project
```

- `my-project`라는 폴더가 새로 만들어지고, Git 저장소로 초기화됨

---

## 4. `.git/` 내부 디렉토리 구조

`git init`을 실행하면 `.git/` 폴더가 생성됩니다. 이 폴더는 Git이 사용하는 **모든 내부 데이터**를 담고 있어요.

```
.git/
├── HEAD
├── config
├── description
├── hooks/
├── info/
├── objects/
├── refs/
```

각 디렉토리의 역할 요약:

| 항목 | 설명 |
| --- | --- |
| `HEAD` | 현재 브랜치 정보를 담고 있음 |
| `config` | 저장소 설정 파일 |
| `objects/` | 커밋, 블롭(blob) 등의 Git 객체 |
| `refs/` | 브랜치 및 태그 정보 |
| `hooks/` | 커밋, 푸시 등 이벤트 발생 시 실행할 스크립트 (자동화 용도) |

> .git/ 폴더는 절대 삭제하거나 수정하면 안 됩니다! 실수로 삭제하면 히스토리도 삭제됩니다.
> 

---

## 5. 기본 브랜치 이름을 main으로 바꾸기

처음 시작 후 기본 브랜치의 이름을 바꿔야 할 상황이 있습니다. git에서는 master라는 이름의 브랜치를 기본 이름으로 사용하지만, gitHub에서는 main이라는 이름을 기본 브랜치로 사용하기 때문입니다.

Git 2.28 이상부터는 기본 브랜치명을 설정할 수 있어요.

### 방법 1: 글로벌 설정

```bash
git config --global init.defaultBranch main
```

이후부터 생성되는 모든 저장소의 초기 브랜치가 `main`이 됨

---

### 방법 2: 생성 후 이름 변경

```bash
git init       # 기본값 master로 생성됨
git branch -m main
```

→ 기존 `master` 브랜치를 `main`으로 이름 변경

---

## 6. 원격 저장소 등록 및 push

GitHub, GitLab, Bitbucket 등에서 저장소를 먼저 만들고, 로컬 저장소와 연결할 수 있습니다.

### 1️⃣ 원격 저장소 등록

```bash
git remote add origin https://github.com/username/repo.git
```

### 2️⃣ 첫 커밋 후 push

```bash
git add .
git commit -m "Initial commit"
git push -u origin main
```

> -u 옵션은 로컬 브랜치와 원격 브랜치를 연결(tracking)해서 이후에는 git push만으로도 동기화 가능하게 함
> 

---

## 7. 실전 초기화 템플릿

```bash
# 1. 프로젝트 디렉토리 생성
mkdir my-app
cd my-app

# 2. Git 저장소 초기화 및 기본 브랜치 설정
git init
git branch -m main     # 기본 브랜치 이름 변경 (선택)

# 3. GitHub 저장소 연결
git remote add origin https://github.com/username/my-app.git

# 4. 파일 생성 후 첫 커밋
echo "# my-app" > README.md
git add .
git commit -m "Initial commit"

# 5. 원격 저장소로 push
git push -u origin main
```

---

## 8. 주의사항 & 실전 팁

### ✅ 팁

- `git init`은 로컬 저장소만 생성 → 원격 연결은 별도 설정 필요 (`git remote add`)
- 저장소에 불필요한 파일이 추적되지 않도록 **`.gitignore` 파일은 반드시 작성**
- `.git` 폴더가 보이지 않으면 `ls -a` 또는 `dir /a` 명령어로 확인

### ❌ 실수 방지

| 실수 | 해결 방법 |
| --- | --- |
| 실수로 잘못된 위치에서 `git init` 실행 | `.git/` 폴더를 삭제하면 Git 관리 해제 |
| 같은 폴더에서 두 번 이상 `git init` | 중복 실행해도 문제는 없지만 변경 없음 |

---

## 9. 마무리

`git init`은 단순하지만 **모든 Git 프로젝트의 시작점**입니다.

그저 명령어를 한 줄 입력하는 것을 넘어, **Git이 어떤 구조로 작동하고 어떤 파일을 추적하기 시작하는지** 이해하면 이후 브랜치/커밋/원격 작업이 더 쉬워집니다.

또한, **브랜치 전략**, **원격 연결**, **초기 커밋**까지 연결되는 아주 중요한 첫 단계입니다.

프로젝트를 시작할 때마다 매번 설정하던 것들을 템플릿화해두면,

효율도 높아지고 실수도 줄일 수 있어요!

---
