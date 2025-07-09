# 내가 보려고 정리한 Anaconda 명령어

### 콘다 정보 확인

conda info

### 버전 확인

conda --version

### 가상환경 생성

conda create -n 환경명 python=버전

conda create --name 환경명 python=버전

### 리스트 확인

conda env list

### conda prompt clear

cls

### 가상환경 삭제

conda remove -n 환경명 —all

### 가상환경 실행

activate 가상환경명

### 가상환경 종료

conda deactivate

### 깃허브 프로젝트 가져오기

git clone <url>

### 리눅스 도스키 지정

doskey ls = dir

doskey clear = cls

### jupyter notebook kernel 추가

1. pip install ipykernel  로 설치
    
    conda install ipykernel
    
2. python -m ipykernel install —user —name ‘가상환경명’ —display-name ‘표시할 이름’

—user : 현재 사용자 권한 수준으로 kernel 등록

—name : Jupyter Notebook 내부적으로 사용할 가상환경의 명칭이다. activate 된 가상환경 실제 명칭에 상관없이 alias처럼 작동한다.

—display-name : 가상환경에 kernel 등록할 명칭

### kernel 제거

1. jupyter kernelspec list   :  리스트 확인
2. jupyter kernelspec remove 환경명(—name)
