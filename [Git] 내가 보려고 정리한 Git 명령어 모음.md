# [Git] 내가 보려고 정리한 Git 명령어 모음

git 기본정보 입력

```jsx
git config --global user.name '이름'

git config --global user.email '이메일'
```

### Git 시작

```jsx
git init
```

작업 트리로 사용할 디렉토리에서 입력하면 .git 폴더 생성

### Git 상태 확인

```jsx
git status
```

### 수정한 파일 스테이징 - git add

```jsx
git add '파일명'
```

### 스테이지에 올라온 파일 커밋하기 - git commit

```jsx
git commit -m '입력할 메세지'

git commit -am '입력할 메세지'  # add + commit
```

### 저장소에 저장된 버전 확인, 커밋 기록 확인 - git log

```jsx
git log

git log --stat  # 커밋 메세지와 관련 파일 표시
```

: 로그 화면이 너무 많으면 한 화면씩 나누어 보여줌

: `Enter`로 다음 로그화면을 볼 수 있고

: `Q`를 누르면 로그 화면을 빠져 나와서 다시 깃 명령 입력 가능

### 변경사항 확인 - git diff

```jsx
git diff
```

### 버전 관리 제외하기 - .gitignore

### 최근 커밋 메시지 수정하기

```jsx
git commit --amend  # vim 모드로 변경, 수정
```

### 작업 트리에서 수정한 파일 되돌리기 - git checkout

수정한 내용을 취소하고 최신 버전 상태로 되돌리기

```jsx
git checkout -- '파일명'

git restore '파일명'
```

### 스테이징 되돌리기 - git reset HEAD 파일명

스테이지에 올라간 (add 명령 이후) 파일 내리기

```jsx
git reset HEAD '파일명'

git restore --staged '파일명'
```

### 커밋 되돌리기

```jsx
git reset HEAD^

# 최근 3개 커밋 되돌리기
git reset HEAD~3
```

**git reset 명령의 옵션**

`--soft HEAD^` : 최근 커밋 하기 전 상태로 작업트리 되돌리기

`--mixed HEAD^` : 최근 커밋과 스테이징 하기 전 상태로 작업트리 되돌리기, 기본값

`--hard HEAD^` : 최근 커밋과, 스테이징, 파일 수정 하기 전 상태로 작업트리 되돌리기, 복구 불가

### 특정 커밋으로 되돌리기 - git reset 커밋 해시

커밋 해시는 커밋ID 라고도 함

커밋 되돌리기 주의점!!

reset A 라고하면 A가 리셋이 아니라 최근 커밋을 A로 리셋(A 이후 커밋을 삭제)

```jsx
git reset --hard '복사한 커밋 해시'
```

### 커밋 삭제하지 않고 되돌리기  - git revert

커밋을 되돌리더라도 취소한 커밋 남겨두기

revert 명령 뒤에 취소할 커밋 버전 해시 지정

```jsx
git revert '복사한 커밋 해시'
```

: revert를 하면 커밋을 삭제하는 대신 변경 이력을 취소한 REVERT 커밋을 생성함