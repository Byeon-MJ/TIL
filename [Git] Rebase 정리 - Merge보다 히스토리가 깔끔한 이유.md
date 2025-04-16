Git을 조금만 써보면 자연스럽게 등장하는 두 명령어, `merge`와 `rebase`.

지난 포스팅에서는 `merge`를 깊게 다뤘다면, 이번에는 **히스토리를 깔끔하게 정리하는 마법 같은 명령어 `rebase`**를 파헤쳐봅니다.

---

## 📌 목차

1. `git rebase`란?
2. rebase의 작동 원리
3. rebase 사용법
4. rebase vs merge: 히스토리 비교
5. rebase 충돌 처리
6. 인터랙티브 rebase (`i`)
7. 주의사항 및 실전 팁
8. 마무리

---

## 1. `git rebase`란?

`git rebase`는 **한 브랜치의 커밋들을 다른 브랜치의 마지막 커밋 이후로 "재배치(relocation)"하는 명령어**입니다.

즉, 커밋들을 복사해서 새로 붙여 넣는 느낌으로 작동합니다.

> "브랜치 히스토리를 병합(Merge)하는 게 아니라,
> 
> 
> 다른 베이스(기준) 위로 옮기는 것"
> 

---

## 2. 작동 원리

예시 상황:

```
main:    A---B---C
                  \
feature:           D---E
```

이 상태에서 `feature` 브랜치에서 `rebase main`을 하면?

```
main:    A---B---C
                      \
feature (after):       D'---E'
```

- `D`, `E` 커밋이 `C` 위로 새롭게 생성됩니다 (`D'`, `E'`)
- 원래 커밋은 그대로 있지만 새로운 커밋으로 재작성된다는 점이 중요!

---

## 3. Rebase 사용법

### 기본 사용

```bash
git checkout feature
git rebase main
```

이렇게 하면 `feature` 브랜치의 커밋이 `main` 위에 올라가요.

---

### 예: main 브랜치의 최신 내용을 내 브랜치에 반영하고 싶을 때

```bash
git checkout my-feature
git fetch origin
git rebase origin/main
```

---

### 브랜치 병합처럼 쓸 수도 있어요

```bash
git checkout main
git rebase feature  # ❌ 비추 (일반적으로 이렇게 안 씀)
```

> 보통은 작업 브랜치에서 main을 rebase하는 방식이 안전합니다.
> 

---

## 4. Rebase vs Merge: 히스토리 비교

### Merge 결과

```
A---B---C----------M
         \       /
          D---E
```

### Rebase 결과

```
A---B---C---D'---E'
```

| 항목 | Merge | Rebase |
| --- | --- | --- |
| 커밋 히스토리 | 병합 커밋 있음 | 병합 커밋 없음 |
| 그래프 | 브랜치 분기 명확 | 직선형, 깔끔 |
| 협업 추적 | 분기 추적 쉬움 | 커밋 단순화됨 |
| 위험도 | 안전 | 협업 중이면 주의 필요 |

---

## 5. 충돌 발생 시

### 충돌 발생 시 메시지

```bash
error: could not apply <commit hash>
```

### 해결 방법

```bash
# 충돌 해결 후
git add <수정한 파일>
git rebase --continue
```

또는 작업을 취소하고 싶다면:

```bash
git rebase --abort
```

---

## 6. 인터랙티브 Rebase (`i`)

`rebase -i`는 커밋들을 **편집, 삭제, 정렬, 합치기**까지 가능한 강력한 도구입니다.

### 사용 방법

```bash
git rebase -i HEAD~3
```

### 가능한 명령어들

| 명령 | 설명 |
| --- | --- |
| `pick` | 그대로 사용 |
| `reword` | 메시지 수정 |
| `edit` | 커밋 내용 수정 |
| `squash` | 이전 커밋과 합치기 |
| `drop` | 삭제 |

### 예시

```
pick  abc123  기능 A 추가
squash def456  기능 A 마무리
reword ghi789  기능 B 리팩터링
```

---

## 7. 주의사항 및 실전 팁

### 🔒 협업 중인 브랜치에서는 신중하게 사용

- 이미 `push`한 브랜치를 `rebase`하면 **커밋 해시가 바뀌어** 충돌 발생 가능성 있음
- 이럴 땐 `merge`를 사용하는 것이 더 안전

### 💡 실전 팁

- **작업 브랜치 정리용**으로 강력함
- **커밋 로그가 너무 많거나 산만할 때** 사용
- PR 전에 `rebase -i`로 커밋 메시지를 정리하면 매우 깔끔

---

## 8. 마무리

`rebase`는 Git의 역사적 기록을 **다시 쓰는 도구**입니다.

위험할 수도 있지만, 제대로만 쓰면 **깔끔한 히스토리 관리의 핵심 무기**입니다.
