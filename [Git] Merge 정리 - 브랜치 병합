Git에서 `merge`는 **다른 브랜치의 변경사항을 현재 브랜치에 통합**할 때 사용하는 명령어입니다. 협업, 기능 개발, 릴리즈 브랜치 병합 등에서 가장 자주 쓰이는 명령 중 하나입니다.

---

## 📌 목차

1. `git merge`란?
2. Fast-forward 병합 vs 3-way 병합
3. Merge 사용법
4. Merge 충돌(Conflict) 처리
5. Merge 옵션 정리
6. 실제 사용 시나리오
7. 장단점 정리
8. 마무리 & 팁

---

## 1. `git merge`란?

`git merge`는 **두 개의 브랜치를 하나로 합치는 명령어**입니다. 보통 `main` 브랜치로 기능 브랜치를 병합하거나, 반대로 `main`의 최신 내용을 기능 브랜치에 반영할 때 사용합니다.

> **기억할 점**
> 
> 
> 병합의 대상은 항상 "현재 체크아웃된 브랜치"이며, 다른 브랜치를 **가져오는 것**
> 

---

## 2. Fast-forward vs 3-way Merge

### Fast-forward 병합

- **브랜치가 직선으로 이어질 수 있을 때**
- 별도의 병합 커밋이 생성되지 않음
- 커밋 히스토리가 깔끔함

```
main:    A---B
feature:     \
              C---D  (병합 대상)

=> 결과:
main:    A---B---C---D
```

```bash
git checkout main
git merge feature
```

### 3-way 병합 (일반적인 Merge)

- **두 브랜치가 동시에 발전한 경우**
- Git은 공통 조상(commit)을 기준으로 세 방향의 변경사항을 병합
- 병합 커밋(Merge Commit)이 생성됨

```
main:    A---B---C
                 \
feature:          D---E

=> 결과:
main:    A---B---C--------M
                 \      /
                  D---E

```

```bash
git checkout main
git merge feature
```

---

## 3. Git Merge 사용법

### 기본 문법

```bash
git checkout <병합할 브랜치>
git merge <가져올 브랜치>
```

예시:

```bash
git checkout main
git merge feature
```

### 병합 후 자동 커밋

- Git은 변경 사항을 합친 후 자동으로 병합 커밋을 생성합니다.
- 충돌이 없다면 바로 완료됩니다.

---

## 4. Merge 충돌 처리

변경사항이 겹치는 경우 충돌(conflict)이 발생합니다.

### 충돌 발생 시 흐름

```bash
git merge feature
# 충돌 발생
# >>>>>> HEAD 와 <<<<<< feature 표시 확인
```

### 충돌 해결 후

```bash
# 충돌 해결하고 저장한 후
git add <수정한 파일>
git commit
```

또는 Git이 직접 메시지를 제안함:

```bash
git merge --continue
```

---

## 5. Git Merge 옵션

| 옵션 | 설명 |
| --- | --- |
| `--no-ff` | Fast-forward 병합을 방지하고 병합 커밋 생성 |
| `--squash` | 커밋을 하나로 압축(squash)하고, 직접 커밋 |
| `--abort` | 충돌 발생 시 병합 중단 |
| `--commit` | 자동 커밋 활성화 (기본값) |
| `--no-commit` | 병합만 하고 직접 커밋하도록 함 |

예:

```bash
git merge feature --no-ff
```

---

## 6. 실제 사용 시나리오

### 기능 브랜치를 main에 병합

```bash
git checkout main
git merge feature
```

### main 브랜치의 최신 내용을 내 작업 브랜치로 병합

```bash
git checkout feature
git merge main
```

### 병합 커밋을 만들지 않고 squash 병합

```bash
git checkout main
git merge --squash feature
git commit -m "Feature 기능 전체 병합"
```

---

## 7. Merge의 장단점

### 장점

- 브랜치의 변경사항을 안전하게 통합
- 히스토리에 병합 내역이 남아 협업 시 추적 가능
- 충돌이 명확하게 드러나 해결 용이

### 단점

- 히스토리가 복잡해질 수 있음
- 병합 커밋이 많아지면 로그 추적이 어려워짐
- 개인 브랜치에서 지나치게 merge 남용 시 커밋 기록이 지저분해짐

---

## 8. 마무리 & 실전 팁

- 협업 중이라면 **merge를 우선적으로 고려**하자
- 개인 작업 중이라면 `rebase`로 커밋을 정리한 후 merge하면 깔끔
- `-no-ff` 옵션을 사용하면 **기능 브랜치의 존재를 기록**할 수 있어 추적에 유리
- Git 그래프 툴 (`git log --graph`)을 함께 활용하면 merge 히스토리를 더 잘 이해할 수 있음

---
