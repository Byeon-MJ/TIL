# 4.2 컬러 스페이스

영상에 색상과 명암을 표현하는 방법들, 각각의 차이 그리고 활용 방법에 대한 공부

### 4.2.1 디지털 영상의 종류

디지털화된 이미지는 픽셀(pixel)이라는 단위가 여러 개 모여서 그림을 표현한다.

하나의 픽셀을 어떻게 구성하느냐에 따라 이미지를 구분

**바이너리(binary, 이진) 이미지**

- 한 개의 픽셀을 두 가지 값으로만 표현한 이미지
- 두 가지 값은 0과 1 또는 0과 255를 사용한다. 보통 0은 검은색 1이나 255는 흰색을 의미
- 값으로는 명암을 표현할 수 없고, 점의 밀도로 명암을 표현
- 영상 작업에서 피사체의 색상과 명암 정보는 필요 없고 오직 피사체의 모양 정보만 필요할 때 이런 이미지를 사용

**그레이 스케일 이미지**

- 흔히 우리가 흑백 사진이라고 하는 것
- 한 개의 픽셀을 **0~255**의 값으로 표현하고, 픽셀 값의 크기로 명암을 표현
- 한 픽셀이 가질 수 있는 값이 0~255이므로 **부호 없는 1바이트의 크기(uint8)**로 표현하는 것이 일반적
- 이미지 프로세싱에서는 색상 정보가 쓸모 없을 때 컬러 이미지의 색상 정보를 제거함으로써 연산의 양을 줄이기 위해 그레이 스케일 이미지를 사용

**컬러 이미지**

- 컬러 이미지는 한 픽셀당 0~255의 값 3개를 조합해서 표현
- **컬러 스페이스(color space)** : 각 바이트마다 어떤 색상 표현의 역할을 맡을지를 결정하는 시스템
- 컬러 스페이스의 종류는 **RGB, HSV, YUV(YCbCR), CMYK** 등 다양하다.

### 4.2.2 RGB, BRG, RGBA

- 컴퓨터로 이미지에 색상을 표현하는 방법 중 가장 많이 사용하는 방법
- 빛의 3원소인 빨강, 초록, 파랑 세 가지 색의 빛을 섞어서 원하는 색을 표현한다.
- 각 색상은 0~255 범위로 표현, 값이 커질수록 해당 색상의 빛이 밝아지는 원리
- 색상의 값이 모두 255일 때는 흰색, 모든 색상 값이 0일 때는 검은색이 표현
- RGB 이미지는 3차원 배열로 표현( **row x column x channel )**
- OpenCV는 BGR의 순서를 사용한다.

[Why does OpenCV use BGR color format ? | LearnOpenCV #](https://learnopencv.com/why-does-opencv-use-bgr-color-format/)

- BGRA는 배경을 투명 처리하기 위해 알파(alpha) 채널을 추가한 것, 배경의 투명도를 표현하기 위해서 0과 255만을 사용하는 경우가 많다.
- cv2.imread() 함수의 두 째 인자 mode_flag가
    - cv2.IMREAD_COLOR → BGR 이미지
    - cv2.IMREAD_UNCHANGED → 대상 이미지가 알파 채널을 가지고 있다면 BGRA

> **[예제 4-5] BGR, BGRA, Alpha 채널(4.5_BGR_BGRA_Alpha.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> img = cv2.imread('./img/opencv_logo.png')
> 
> # IMREAD_COLOR
> bgr = cv2.imread('./img/opencv_logo.png', cv2.IMREAD_COLOR)
> 
> # IMREADD_UNCHANGED
> bgra = cv2.imread('./img/opencv_logo.png', cv2.IMREAD_UNCHANGED)
> 
> # 이미지 shape
> print('Default:', img.shape)
> print('Color:', bgr.shape)
> print('Unchanged:', bgra.shape)
> 
> >>> Default: (120, 98, 3)
> >>> Color: (120, 98, 3)
> >>> Unchanged: (120, 98, 4)
> 
> cv2.imshow('bgr', bgr)
> cv2.imshow('bgra', bgra)
> cv2.imshow('alpha', bgra[:, :, 3])  # 알파 채널만 표시
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> 
> # matplotlib으로 표현하기 - Alpha 채널이 추가되면 이상하다..?
> bgr_r = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
> bgra_r = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
> 
> fig = plt.figure(figsize=(5, 5))
> 
> plt.subplot(1, 3, 1)
> plt.imshow(bgr_r)
> plt.xticks([]), plt.yticks([])
> 
> # ...???
> plt.subplot(1, 3, 2)
> plt.imshow(bgra_r)
> plt.xticks([]), plt.yticks([])
> 
> # ......??????
> plt.subplot(1, 3, 3)
> plt.imshow(bgra_r[:, :, 3])
> plt.xticks([]), plt.yticks([])
> 
> plt.show()
> ```
> 
> ![Untitled](https://user-images.githubusercontent.com/69300448/220490130-45acb74a-ea39-41f8-91d1-6a4fcfd41d63.png)
> 
- 원본 이미지는 배경이 투명한 OpenCV 로고이다. 왼쪽 두 그림은 투명한 배경이 검은색으로 표시되었고, 로고 아래의 글씨도 원래 검은색이라서 보이지 않는다.
- cv2.IMREAD_UNCHANGED 옵션으로 읽은 이미지는 shape가 (240, 195, 4)로 마지막 채널이 하나 더 있는 것을 알 수 있다.
- 마지막 알파 채널만 표시해보니 로고와 글씨를 제외하고는 모두 검은색으로 표시된다.
- 전경은 255, 배경은 0의 값을 가지게 됨, 이를 이용해 전경과 배경을 손쉽게 분리 할 수 있다.

### 4.2.3 컬러 스페이스 변환

컬러 이미지를 그레이 스케일로 변환하는 것은 이미지 연산의 양을 줄여서 속도를 높이는데 꼭 필요

그레이 스케일이나 다른 컬러 스페이스로 변환하는 방법

1. 직접 알고리즘을 구현
2. OpenCV에서 제공하는 cv2.cvtColor() 함수를 이용

**3채널의 평균값을 구해서 그레이 스케일로 변환하는 방법**

> **[예제 4-6] BGR을 그레이 스케일로 변환(4.6_BGR2GRAY.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/girl.jpg')
> 
> # dtype 변경
> img2 = img.astype(np.uint16)
> 
> # 채널별로 분리
> b, g, r = cv2.split(img2)
> 
> # 평균값 연산 후 dtype 되돌리기
> gray1 = ((b + g + r)/3).astype(np.uint8)
> 
> # BGR을 그레이 스케일로 변경
> gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> 
> cv2.imshow('Original', img)
> cv2.imshow('Gray1', gray1)
> cv2.imshow('Gray2', gray2)
> 
> cv2.waitKey(0)
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 1](https://user-images.githubusercontent.com/69300448/220490156-9f4fbf79-8717-4d7f-a944-f38157f350bb.png)
> 

위 예제에서 dtype을 uint16 타입으로 변경한 이유는 평균값을 구하는 과정에서 3채널의 값을 합했을 때 255보다 큰 값이 나오면 uint8로는 표현하기 불가능하므로 변경하여 계산을 마친 후 다시 uint8로 되돌리는 방법을 사용하였다.

**cv2.split() 함수** : 매개변수로 전달한 이미지를 채널별로 분리해서 튜플로 반환, 아래 Numpy 슬라이싱과 동일하다.

```python
b, g, r = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2
```

컬러 이미지를 그레이 스케일로 변환할 때 정확한 명암을 얻으려면 단순히 평균값만 계산하는 것보다 더 정교한 연산이 필요하다. 컬러 스케일 간의 변환은 컬러 스케일을 그레이 스케일로 바꾸는 것보다 더 복잡한 알고리즘이 필요, OpenCV에서는 **cv2.cvtColor(img, flag)** 함수를 제공한다.

- **out = cv2.cvtColor(img, flag)**
    - img : Numpy 배열, 변환할 이미지
    - flag : 변환할 컬러 스페이스, **cv2.COLOR_** 로 시작하는 이름(총 274개)
        - cv2.COLOR_BGR2GRAY : BGR 컬러 이미지를 그레이 스케일로 변환
        - cv2.COLOR_GRAY2BGR : 그레이 스케일 이미지를 BGR 컬러 이미지로 변환
        - cv2.COLOR_BGR2RGB : BGR 컬러 이미지를 RGB 컬러 이미지로 변환
        - cv2.COLOR_BGR2HSV : BGR 컬러 이미지를 HSV 컬러 이미지로 변환
        - cv2.COLOR_HSV2BGR : HSV 컬러 이미지를 BGR 컬러 이미지로 변환
        - cv2.COLOR_BGR2YUV : BGR 컬러 이미지를 YUV 컬러 이미지로 변환
        - cv2.COLOR_YUV2BGR : YUV 컬러 이미지를 BGR 컬러 이미지로 변환
    - out : 변환된 결과 이미지(Numpy 배열)

**cv2.COLOR_GRAY2BGR**의 경우, 그레이 스케일로 BGR 스케일로 변환하는데, 실제로 흑백 사진을 컬러 사진으로 바꿔주는 것이 아니다! 2차원 배열 이미지를 3개 채널이 모두 같은 값을 가지는 3차원 배열로 변환하는 것. 영상 간 연산을 할 때, 서로 차원이 다르면 연산을 할 수 없기 떄문에 차원을 맞추는 용도로 사용한다.

### 4.2.4 HSV, HSI, HSL

- RGB와 마찬가지로 3채널로 컬러 이미지를 표시
- **H(Hue, 색조), S(Saturation, 채도), V(Value, 명도)**
- 마지막 V 대신 **I(Intensity, 밀도), L(Lightness, 명도)**로 표기하기도 한다.
- H 값은 픽셀이 어떤 색인지를 표현한다.
    - 색상에 매칭되는 숫자를 매겨놓고 그 360도 범위의 값을 갖게 해서 색을 표현
    - OpenCV에서 영상을 표현 할때 최대 값이 255를 넘지 못하므로 360을 반으로 나누어 0~180 범위의 값으로 표현하고 180보다 큰 값은 180으로 간주
    - 빨강(165~180, 0~15), 초록(45~75), 파랑(90~120)
- S 값은 채도, 포화도, 또는 순도이다. 해당 색상이 얼마나 순사하게 포함되어 있는지를 표현
    - S값은 0~255 범위로 표현, 255는 가장 순수한 색상
- V 값은 명도, 빛이 얼마나 밝은지 어두운지를 표현하는 값
    - 범위는 0~255, 255가 가장 밝은 상태, 0은 가장 어두운 상태로 검은색이 표시
- HSV 포맷은 오직 H 채널 값만 확인하면 되므로 색상을 기반으로 하는 여러가지 작업에 효과적
    
    

> **[예제 4-7] BGR에서 HSV로 변환(4.7_BGR2HSV.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> # BGR 컬러 스페이스로 원색 픽셀 생성
> red_bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)      # 빨강
> green_bgr = np.array([[[0, 255, 0]]], dtype=np.uint8)    # 초록
> blue_bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)     # 파랑
> yellow_bgr = np.array([[[0, 255, 255]]], dtype=np.uint8) # 노랑.
> 
> # BGR to HSV
> red_hsv = cv2.cvtColor(red_bgr, cv2.COLOR_BGR2HSV)
> green_hsv = cv2.cvtColor(green_bgr, cv2.COLOR_BGR2HSV)
> blue_hsv = cv2.cvtColor(blue_bgr, cv2.COLOR_BGR2HSV)
> yellow_hsv = cv2.cvtColor(yellow_bgr, cv2.COLOR_BGR2HSV)
> 
> # HSV 픽셀 출력
> print('red:', red_hsv)
> print('green:', green_hsv)
> print('blue:', blue_hsv)
> print('yellow:', yellow_hsv)
> 
> >>> Red: [[[  0 255 255]]]
> >>> Green: [[[ 60 255 255]]]
> >>> Blue: [[[120 255 255]]]
> >>> Yellow: [[[ 30 255 255]]]
> ```
> 

### 4.2.5 YUV, YCbCr

- YUV 포맷은 사람이 색상을 인식할 때 밝기에 더 민감하고 색상은 상대적으로 둔감한 점을 고려해서 만든 컬러 스페이스
- **Y**는 밝기(Luma), **U(Chroma Blue, Cb)**는 밝기와 파란색과의 색상 차, **V(Chroma Red, Cr)**는 밝기와 빨간색과의 색상 차를 표현
- Y에 많은 비트수를 할당하고 U와 V에는 적은 비트수를 할당해서 데이터를 압축하는 효과
- YUV는 YCbCr 포맷과 혼용되기도 한다.
    - 본래 YUV는 본래 텔레비전 시스템에서 아날로그 컬러 정보를 인코딩하는데 사용
    - YCbCr 포맷은 MPEG나 JPEG와 같은 디지털 컬러 정보를 인코딩하는데 사용
    - 요즘들어 YUV는 YCbCr로 인코딩된 파일 포맷을 설명하는 용어로 사용됨
- YUV는 밝기 정보와 컬러 정보를 분리해서 사용하므로 명암대비(contrast)가 좋지 않은 영상을 좋게 만드는 데 활용된다.

> **[예제 4-8] BGR에서 YUV로 변환(4.8_BGR2YUV.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> # BGR 컬러 스페이스로 세 가지 밝기의 픽셀 생성
> dark = np.array([[[0, 0, 0]]], dtype=np.uint8)
> middle = np.array([[[127, 127, 127]]], dtype=np.uint8)
> bright = np.array([[[255, 255, 255]]], dtype=np.uint8)
> 
> # BGR to YUV
> dark_yuv= cv2.cvtColor(dark, cv2.COLOR_BGR2YUV)
> middle_yuv= cv2.cvtColor(middle, cv2.COLOR_BGR2YUV)
> bright_yuv= cv2.cvtColor(bright, cv2.COLOR_BGR2YUV)
> 
> # YUV 변환 픽셀 출력
> print('Dark:', dark_yuv)
> print('Middle:', middle_yuv)
> print('Bright:', bright_yuv)
> 
> >>> Dark: [[[  0 128 128]]]
> >>> Middle: [[[127 128 128]]]
> >>> Bright: [[[255 128 128]]]
> ```
> 

출력 결과에서 보면 밝기 정도는 Y 채널에만 나타나는 것을 알 수 있다.

픽셀의 밝기를 제어해야 할 때, BGR 포맷은 3채널을 모두 연산해야 한다. 하지만 YUV 포맷은 Y 채널 하나만 작업하면 되므로 효과적이다.
