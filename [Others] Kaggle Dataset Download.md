# [Others] Kaggle 데이터셋 다운로드

딥러닝 실습을 할 때 모델 학습을 위한 데이터셋을 받기 위해 여러가지 방법을 사용할 수 있다.

데이터를 직접 다운 받아서 사용할 수도 있지만 Kaggle API를 사용하면 조금 더 편하게 데이터를 받아서 사용할 수 있다.

오늘은 Kaggle API를 활용하여 Kaggle 데이터셋을 직접 다운로드 하는 방법을 포스팅 하려한다.

# 1. Kaggle 패키지 설치하기

Kaggle API Token 을 다운 받았으면 패키지 설치를 통해 Kaggle API를 사용할 수 있다.

```python
# Python3
!pip3 install kaggle

# Colab -> Colab은 이미 설치되어 있어서 패스 가능
!pip install kaggle

# Anaconda
conda install kaggle
```

# 2. Kaggle API Token 준비하기

## Kaggle API Token 다운로드

먼저 Kaggle API를 사용하기 위해 Token을 준비해야한다.

Kaggle에서 **`Account`**에 들어가면 API Token을 생성할 수 있는 부분이 있다.

**`Create API Token`** 버튼을 눌러서 **`kaggle.json`** 파일을 다운받아준다.

![Untitled](%5BOthers%5D%20Kaggle%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%89%E1%85%A6%E1%86%BA%20%E1%84%83%E1%85%A1%E1%84%8B%E1%85%AE%E1%86%AB%E1%84%85%E1%85%A9%E1%84%83%E1%85%B3%203d6e9af9777e4d6d93fbd42bd01189f3/Untitled.png)

![Untitled](%5BOthers%5D%20Kaggle%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%89%E1%85%A6%E1%86%BA%20%E1%84%83%E1%85%A1%E1%84%8B%E1%85%AE%E1%86%AB%E1%84%85%E1%85%A9%E1%84%83%E1%85%B3%203d6e9af9777e4d6d93fbd42bd01189f3/Untitled%201.png)

# 3. kaggle.json 업로드, 이동하기

나는 Colab 환경에서 분석을 진행하였기 때문에 Colab 코드를 이용하였다.

아래 코드를 통해 kaggle.json 파일을 업로드 한다.

```python
from google.colab import files
files.upload()
```

그 다음 kaggle.json 파일을 아래 명령어를 통해 이동시켜준다.

```python
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

# 4. 데이터 다운로드

필요한 데이터셋이 있는 페이지로 이동한 뒤, `Copy API command` 버튼을 이용해 API 코드를 복사하고 아래 코드처럼 붙여넣은 후 실행하면 데이터셋이 다운로드가 된다.

![Untitled](%5BOthers%5D%20Kaggle%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%89%E1%85%A6%E1%86%BA%20%E1%84%83%E1%85%A1%E1%84%8B%E1%85%AE%E1%86%AB%E1%84%85%E1%85%A9%E1%84%83%E1%85%B3%203d6e9af9777e4d6d93fbd42bd01189f3/Untitled%202.png)

```python
!kaggle kernels output nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset -p /path/to/dest
```