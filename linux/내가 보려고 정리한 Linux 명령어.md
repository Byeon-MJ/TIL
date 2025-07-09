# 내가 보려고 정리한 Linux 명령어

# **Linux 디렉토리 관련**

- 디렉토리는 root라고 하는 '/'로 시작한다.
- 디렉토리 이름 '.'은 현재 위치 이다.
- 디렉토리 이름 '..'은 상위 디렉토리 이다

# **Linux 명령어**

- [Python](https://www.notion.so/Linux-Linux-25cdb68b84e04258a4a4f9125c3e9861)
    - 실행 : python
    - 라이브러리 설치 : pip
- [디렉토리/파일](https://www.notion.so/Linux-Linux-25cdb68b84e04258a4a4f9125c3e9861)
    - 디렉토리 위치 이동 : cd
    - 현재 디렉토리 보기 : pwd
    - 파일 내용 보기 : ls
    - 디렉토리 생성 : mkdir
    - 디렉토리/파일 삭제 : rm
    - 디렉토리/파일 카피 : cp
    - 디렉토리/파일 이동 : mv
- [파일 보기](https://www.notion.so/Linux-Linux-25cdb68b84e04258a4a4f9125c3e9861)
    - 파일 전체 보기 : cat
    - 파일 앞부분 보기 : head
    - 파일 뒷두분 보기 : tail
- [다운로드](https://www.notion.so/Linux-Linux-25cdb68b84e04258a4a4f9125c3e9861)
    - git 프로젝트 다운로드 : git
    - 파일 다운로드 : wget
- [압축](https://www.notion.so/Linux-Linux-25cdb68b84e04258a4a4f9125c3e9861)
    - tar 파일 압출 풀기 : tar
    - rar 파일 압축 풀기 : unrar
    - zip 파일 압축 풀기 : unzip
- [기타](https://www.notion.so/Linux-Linux-25cdb68b84e04258a4a4f9125c3e9861)
    - 프로그램 설치 : apt
    - 파일 내용으로 검색 : grep
    - 갯수 세기 : wc
    - 디렉토리 구조 보기 : tree

## Python

### 실행

```python
# 스크립트 my_script.py 실행
$ python my_script.py
```

### pip

```python
# 라이브러리 some 설치
$ pip install some
```

## 디렉토리/파일

### 디렉토리 위치 이동 : cd

- change directory
- .  : 현재 위치
- .. : 상위 디렉토리
- / : root 디렉토리

```python
# 홈 디렉토리로 이동
$ cd

# 상위 디렉토리로 이동
$ cd ..

# 하위 path로 이동
$ cd path

# 하위 path1/path2로 이동
$ cd path1/path2
```

### 현재 디렉토리 보기 : pwd

- print working directory

```python
# 현재의 디렉토리
$ pwd
/some/path
```

### 파일 내용 보기 : ls

- list

```python
# 현재 디렉토리 내용을 이름만 보기
$ ls

# 현재 디렉토리 내용을 자세히 보기
$ ls -al

# path1 디렉토리 내용을 자세히 보기
$ ls -al path1

# path1/path2 디렉토리 내용을 자세히 보기
$ ls -al path1/path2

# /some/absolute/path 디렉토리 내용을 자세히 보기
$ ls -al /some/absolute/path

# 현재 디렉토리에 있는 .txt로 끝나는 파일과 디렉토리 들을 자세히 보기
$ ls -al *.txt로

# path 디렉토리 밑에 있는 data로 시작하고 .csv로 끝나는 파일과 디렉토리 들을 자세히 보기
$ ls -al path/data*.csv
```

ls -al 의 경우 alias를 지정하여 ll 로도 많이 사용한다.

### 디렉토리 생성 : mkdir

- 옵션
    
    -p(parent) : 전체 경로가 없으면 하나씩 다 생성
    

```python
# tmp 디렉토리 생성
$ mkdir tmp

# path1/path2 디렉토리 생성
$ mkdir -p path1/path2
```

### 디렉토리/파일 삭제 : rm

- 옵션
    
    -r(recursive) : 하위 파일까지 삭제
    
    -f(force) : 강제로 삭제
    

```python
# 파일 file1.txt를 삭제한다. Y/n를 대답해야 한다.
$ rm file1.txt

# 파일 file1.txt를 무조건 삭제한다.
$ rm -rf file1.txt

# 디렉토리 path1을. Y/n를 대답해야 한다. 디렉토리에 무언가 있으면 실패한다.
$ rm path1

# 디렉토리 path1을 무조건 삭제한다.
$ rm -rf path1
```

### 디렉토리/파일 복사 : cp

- 디렉토리를 카피할때는 '-r' 옵션을 넣어주면 카피 가능

```python
# 현재 위치의 file1.txt를 복사하여 file2.txt를 생성한다.
$ cp file1.txt file2.txt

# 현재 위치의 file1.txt를 path 디렉토리 밑에 카피한다.
$ cp file1.txt path/

# 현재 위치의 path1 디렉토리의 모든 것을 카피하여 path2를 생성한다.
$ cp -r path1 path2

# 현재 위치의 path1 디렉토리의 모든 것을 path2밑에 카피한다.
$ cp -r path1 path2/
```

### 디렉토리/파일 이동 : mv

- 파일 이동 및 파일 이름 변경에 사용

```python
# 현재 위치의 file1.txt의 이름을 file2.txt로 변경한다.
$ mv file1.txt file2.txt

# 현재 위치의 file1.txt를 path 디렉토리 밑으로 이동한다.
$ mv file1.txt path/

# 현재 위치의 디렉토리 path1의 이름을 path2로 변경한다.
$ mv path1 path2

# 현재 위치의 디렉토리 path1을 path2 디렉토리 밑으로 이동한다.
$ mv path1 path2/
```

## 파일 보기

### 파일 전체 보기 : cat

```python
# 파일 내용 전부 다 보기
$ cat file.txt
```

### 파일 앞부분 보기 : head

- 기본값으로 10줄을 보여준다.

```python
# file.txt의 앞 10줄 보기
$ head file.txt
$ head -10 file.txt
```

### 파일 뒷부분 보기 : tail

- 기본값으로 10줄을 보여준다.

```python
# file.txt의 뒷 10줄 보기
$ tail file.txt
$ tail -10 file.txt
```

## 다운로드

### git 프로젝트 다운로드 : git

```python
# github의 프로젝트 전체 다운로드
$ git clone http://github.com/some_project
```

![Untitled](https://user-images.githubusercontent.com/69300448/216914162-a13bf381-cd38-4b3a-bb96-766a424b2bda.png)

### 파일 다운로드 : wget

```python
# 파일 1개를 다운로드
$ wget http://some.com/path/file_name
```

- 다운로드 링크 획득하기

    ![Untitled 1](https://user-images.githubusercontent.com/69300448/216914195-9ec8d3ee-721e-45ea-9d09-9c4e4595af15.png)
 
## 압축

### tar 파일 압축 풀기/하기 : tar

```python
# 압축된 tar.gz 파일을 풀기
$ tar xvfz file.tar.gz

# tar 파일을 풀기
$ tar xvf file.tar

# tar로 묶기
$ tar cvf dir.tar dir

# tar.gz 로 압축 하기
$ tar cvfz dir.tar.gz dir
```

- 리눅스에서는 압축파일 형식으로 tar.gz가 많이 사용됨

![Untitled 2](https://user-images.githubusercontent.com/69300448/216914287-51992813-e458-4fbd-b089-885c7e424f85.png)

### rar 파일 압축 풀기/하기 : unrar, rar

```python
# rar 파일 풀기
$ unrar x file.rar

# raw 압축 하기
$ rar a -r dir.rar dir
```

### zip 파일 압축 풀기/하기 : unzip, zip

```python
# zip 파일 풀기
$ unzip file.zip

# zip으로 압축 하기
$ zip -r dogs-cat.zip dogs-cat
```

## 기타

### 프로그램 설치 : apt

- 리눅스 표준 설치 툴

```python
# apt 정보 업데이트
$ apt-get update

$ 프로그램 설치
$ apt inatll git
$ apt inatll tree
```

### 파일 내용으로 검색 : grep

```python
# 현재 디렉토리에서 .txt로 끝나는 파일중에 hi가 포함된 부분 찾기
$ grep hi *.txt

# 현재 디렉토리(.)와 밑의 모든 파일에서 hi가 포함된 부분 찾기
$ grep -r hi .
```

### 갯수 세기 : wc

```python
# 특정 파일의 줄수 세기
$ wc some_file.txt

# 현 디렉토리의 파일 수 세기
$ ls -al | wc
```

### 디렉토리 구조 보기 : tree

```python
# 디렉토리 구조와 파일들 보기
$ tree

# 디렉토리 구조만 보기
# tree -d
```

- tree 사용 전 설치 필요
    
    ```python
    $ apt install tree
    ```
    
- tree 명령어 확인
    
    -d 는 현재 디렉토리에서만
    
    ![Untitled 3](https://user-images.githubusercontent.com/69300448/216914329-a581815c-19a1-4209-afe2-0c1608e87a6e.png)

### Vim 으로 메모장 생성

vim test.txt

ex 모드 - 입력 모드 (i / esc)

입력모드에서 :wq  > 저장 후 종료(write, quit)

:w , :write - 문서 저장

:q, :quit - 편집기 종료

:wq (파일) - 저장하고 종료

:q! - 저장하지않고 종료

cat test.txt - 텍스트 파일 내용 확인

cat 파일 - 파일 내용 표시

cat 파일1, 파일2 … 파일n > 새파일 - 파일 n개를 차례로 연결해서 새 파일 생성

cat 파일1 >> 파일2 - 파일1의 내용을 파일2의 끝에 연결
