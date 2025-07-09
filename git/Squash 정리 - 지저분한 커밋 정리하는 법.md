기능 개발을 하다 보면 커밋이 자잘하게 쌓여서 지저분해지는 경우가 많습니다.

예를 들어, `add 기능 구현`, `fix 오타`, `test 수정`, `README 추가` 등 수많은 커밋들...

이럴 때 등장하는 강력한 커밋 정리 도구가 바로 **Git Squash**입니다.

이번 포스팅에서는 `squash`의 개념, 사용법, 실제 사례, 시각적 흐름, 주의사항까지 전부 정리해드립니다.

---

## 📌 목차

1. `git squash`란?
2. squash가 왜 필요한가?
3. squash 사용법 (rebase 활용)
4. squash 사용법 (merge 활용)
5. 시각적 흐름 예시
6. 실전 사례
7. 주의사항 및 실전 팁
8. 마무리

---

## 1. `git squash`란?

Git에서 **`squash`는 여러 개의 커밋을 하나로 합치는 작업**을 의미합니다.

> squash는 "눌러서 뭉개다"라는 뜻 → 여러 커밋을 하나로 압축!
> 

### 주의: `squash`는 명령어가 아니라 `rebase -i`나 `merge --squash`에서 **동작 방식**으로 쓰입니다.

---

## 2. squash가 왜 필요한가?

- 개발 중 생긴 **자잘한 커밋들을 하나의 의미 있는 커밋으로 정리**하고 싶을 때
- 코드 리뷰 전 커밋 내역을 깔끔하게 만들고 싶을 때
- 기능 단위로 커밋 로그를 관리하고 싶을 때

---

## 3. squash 사용법 (with `git rebase -i`)

가장 일반적인 squash 사용 방식입니다.

### 예제

```bash
git rebase -i HEAD~3
```

### 편집 화면:

```
pick  a1b2c3  기능 A 구현
pick  d4e5f6  console.log 제거
pick  789abc  오타 수정
```

👉 이렇게 수정:

```
pick  a1b2c3  기능 A 구현
squash  d4e5f6  console.log 제거
squash  789abc  오타 수정
```

### 메시지 통합 화면:

```
# 이 커밋 메시지를 편집하세요:
기능 A 구현

console.log 제거
오타 수정
```

→ 메시지를 정리해서 하나의 커밋으로 만들 수 있습니다!

---

## 4. squash 사용법 (with `git merge --squash`)

기능 브랜치 전체를 하나의 커밋으로 병합할 때 사용합니다.

```bash
git checkout main
git merge --squash feature
git commit -m "Feature 기능 통합"
```

### 특징

- 브랜치 내용은 병합되지만
- **병합 커밋은 하나만 생기고 히스토리는 남지 않음**

---

## 5. 시각적 흐름 예시

### Before

```
A---B---C         (main)
         \
          D---E---F  (feature)
```

### squash (via rebase)

```
A---B---C---G     (main or feature)
             ↑
          (D~F 하나로 압축)
```

---

## 6. 실전 사례

### ✅ 기능 개발 중 5개의 커밋을 하나로 만들고 싶을 때

```bash
git rebase -i HEAD~5
```

- 오타 수정, 디버깅용 커밋 등을 하나로 묶고
- 커밋 메시지를 깔끔하게 정리

### ✅ 기능 브랜치를 병합하면서 커밋은 하나로 정리하고 싶을 때

```bash
git checkout main
git merge --squash feature
git commit -m "Feature 추가"
```

- 팀원 리뷰 전에 정리하는 데 적합

---

## 7. 주의사항 및 실전 팁

### 협업 중 squash 시 주의

- 커밋 히스토리를 변경하므로 **이미 푸시한 커밋을 squash하면 충돌 위험**이 있음
- 이럴 땐 **`force push`*가 필요하고, 다른 사람의 브랜치에 영향을 줄 수 있음

### 실전 팁

- squash는 **PR 전에 커밋 정리**할 때 아주 유용
- `fix`, `refactor`, `WIP` 같은 커밋은 squash로 묶는 습관 들이기
- `git log --oneline`으로 결과를 쉽게 확인 가능

---

## 8. 마무리

`squash`는 커밋 히스토리를 깔끔하게 정리할 수 있는 매우 강력한 도구입니다.

협업 시에도, 개인 작업 시에도 **한눈에 이해되는 커밋 로그**를 만드는 데 큰 도움이 되죠.

다만, **이미 푸시된 커밋을 squash할 땐 항상 주의**가 필요합니다.

---
