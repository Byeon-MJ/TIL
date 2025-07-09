# 4.4 이미지 연산

이미지와 영상에서 연산하는 방법을 알아보자.

연산 결과는 새로운 영상을 만들어 내므로 그 자체가 목적이 될 수도 있고, 정보를 얻기 위한 과정일 수도 있다.

### 4.4.1 영상과 영상의 연산

영상에 연산을 할 수 있는 방법은 Numpy의 브로드캐스팅 연산을 직접 적용하는 방법과 OpenCV에서 제공하는 함수를 사용하는 방법이 있다.

영상에서 한 픽셀이 가질 수 있는 값의 범위는 0~255인데, OpenCV의 함수를 사용하면 그 범위를 초과한 결과에 대해서도 안전하게 처리가 가능하다.

- **dest = cv2.add(src1, src2[, dest, mask, dtype])** : src1과 src2 더하기
    - src1 : 입력 영상 1 또는 수
    - src2 : 입력 영상 2 또는 수
    - dest : 출력 영상
    - mask : 0이 아닌 픽셀만 연산
    - dtype : 출력 dtype
- **dest = cv2.substract(src1, src2[, dest, mask, dtype])** : src1 에서 src2를 빼기
    - 모든 인자는 cv2.add() 함수와 동일
- **dest = cv2.multiply(src1, src2[, dest, scale, dtype])** : src1과 src2를 곱하기
    - scale : 연산 결과에 추가 연산할 값
- **dest = cv2.divide(src1, src2[, dest, scale, dtype])** : src1을 src2로 나누기
    - 모든 인자는 cv2.multiply()와 동일

> **[예제 4-13] 영상의 사칙 연산(4.13_arithmetic.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> # 연산에 사용할 배열 생성
> a = np.uint8([[200, 50]])
> b = np.uint8([[100, 100]])
> 
> # Numpy 배열 직접 연산
> add1 = a + b
> sub1 = a - b
> mult1 = a * 2
> div1 = a / 3
> 
> # OpenCV를 이용한 연산
> add2 = cv2.add(a, b)
> sub2 = cv2.subtract(a, b)
> mult2 = cv2.multiply(a, 2)
> div2 = cv2.divide(a, 3)
> 
> # 각 연산 결과 출력
> print(add1, add2)
> print(sub1, sub2)
> print(mult1, mult2)
> print(div1, div2)
> 
> >>> [[ 44 150]] [[255 150]]
> >>> [[100 206]] [[100   0]]
> >>> [[144 100]] [[255 100]]
> >>> [[66.66666667 16.66666667]] [[67 17]]
> ```
> 

출력 결과를 보면, 직접 더한 결과는 255를 초과하여 44이고, **cv2.add()** 함수의 결과는 최대 값인 255이다. 빼기 연산의 결과도 마찬가지로 206으로 정상적이지 않지만, **cv2.subtract()** 함수의 결과는 최소값인 0이 나온다. 곱하기와 나누기도 255를 초과하지 않고 소수점을 갖지 않는 결과가 나온다.

OpenCV의 네 가지 연산을 조금 더 자세히 알아보자. 앞에서 함수에 연산의 대상으로 **두개의 인자**를 전달해 주었다. 그 두 인자의 연산 결과를 **세 번째 인자**로 전달한 배열에 할당하고 결과값으로 반환 받을 수 있다. 만약 **c = a + b와 같은 연산**이 필요하다면 아래와 같은 코드로 적용할 수 있다. 세 코드의 결과는 모두 같다.

```python
c = cv2.add(a, b)
c = cv2.add(a, b, None)
cv2.add(a, b, c)
```

만약 b += a 와 같이 두 입력의 합산 결과를 입력 인자의 하나에 재할당 하고싶을 때는 아래 코드로 작성 할 수 있다.

```python
cv2.add(a, b, b)
b = cv2.add(a, b)
```

만약 **네 번째 인자**인 mask를 지정하는 경우에, 네 번째 인자에 전달한 배열에 어떤 요소 값이 0이면 그 위치의 픽셀은 연산을 하지 않는다.

> **[예제 4-14] mask와 누적 할당 연산(4.14_arithmetic_mask.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> # 연산에 사용할 배열 생성
> a = np. array([[1, 2]], dtype=np.uint8)
> b = np. array([[10, 20]], dtype=np.uint8)
> 
> # 두 번째 요소가 0인 마스크 생성
> mask = np.array([[1, 0]], dtype=np.uint8)# 연산에 사용할 배열 생성
> a = np. array([[1, 2]], dtype=np.uint8)
> b = np. array([[10, 20]], dtype=np.uint8)
> 
> # 두 번째 요소가 0인 마스크 생성
> mask = np.array([[1, 0]], dtype=np.uint8)
> 
> # 누적 할당과의 비교 연산
> c1 = cv2.add(a, b, None, mask)
> print(c1)
> c2 = cv2.add(a, b, b, mask)
> print(c2)
> 
> >>> [[11  0]]
> >>> [[11 20]]
> ```
> 

위 예제에서 보면 a와 b의 더하기 연산은 1+10, 2+20 연산이 각각 이뤄져야 하지만, mask의 두번째 요소의 값이 0이므로 2+20의 연산이 이루어지지 않는다. 따라서 c1의 결과는 [11, 0] 이 된다.

누적 할당을 적용한 c2는 기존의 b에 a가 더해지는데 두 번째 요소는 연산이 무시 되므로 20을 그대로 가져서 [11, 20]이 된다. 이때 주의할 점은 b 자체도 c2와 동일하게 연산의 결과를 갖게 된다. b를 그대로 유지하고 싶다면 아래 코드로 사용할 수 있다.

```python
cv2 = cv2.add(a, b, b.copy(), mask)
```

### 4.4.2 알파 블렌딩

두 영상을 합성하려고 할 때, 더하기 연산이나 cv2.add() 함수만으로는 좋은 결과를 얻기가 어렵다.

직접 더하기 연산을 하면 255를 넘는 경우 초과 값만을 가지게 되므로 영상에서 튀는 색상들이 나타나거나 거뭇거뭇하게 나타나고, cv2.add() 연산을 하면 초과값이 255가 되므로 픽셀 값이 255 가까이 몰리는 현상이 일어나서 하얗게 날아간 것처럼 보인다.

> **[예제 4-15] 이미지 단순 합성(4.15_blending_simple.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 연산에 사용할 이미지 읽기
> img1 = cv2.imread('./img/wing_wall.jpg')
> img2 = cv2.imread('./img/yate.jpg')
> 
> # 이미지 더하기
> img3 = img1 + img2
> img4 = cv2.add(img1, img2)
> 
> imgs = {'img1':img1, 'img2':img2,
>        'img1+img2':img3, 'cv2.add(img1, img2)':img4}
> 
> # 이미지 출력
> for i, (k, v) in enumerate(imgs.items()):
>     plt.subplot(2, 2, i + 1)
>     plt.imshow(v[:, :, ::-1])
>     plt.title(k)
>     plt.xticks([])
>     plt.yticks([])
>     
> plt.show()
> ```
> 
> ![Untitled](https://user-images.githubusercontent.com/69300448/221179327-8c4999f5-96ba-4c3b-8ca6-f2bc7c5477f9.png)
> 

두 영상을 적절하게 합성하려면 각 픽셀의 합이 255가 되지 않게 각각의 영상에 가중치를 줘서 계산을 해야한다. 예를 들어, 두 영상이 정확히 절반씩 반영된 결과 영상을 원한다면 각 영상의 픽셀 값에 각각 50%씩 곱해서 새로운 영상을 생성하면 된다. 각 영상에 적용할 가중치를 **알파(alpha)값**이라고 부른다.

$$
g(x) = (1 - \alpha)f_0(x)\ + \ \alpha f_1(x)
$$

- $f_0(x)$ : 첫 번째 이미지 픽셀 값
- $f_1(x)$ : 두 번째 이미지 픽셀 값
- $\alpha$ : 가중치(알파)
- $g(x)$ : 합성 결과 픽셀 값

OpenCV에서는 이것을 구현한 함수를 제공한다.

- **cv2.addWeight(img1, alpha, img2, beta, gamma)**
    - img1, img2 : 합성할 두 영상
    - alpha : img1에 지정할 가중치(알파 값)
    - beta : img2에 지정할 가중치, 흔히 (1-alpha) 적용
    - gamma : 연산 결과에 가감할 상수, 흔히 0(zero) 적용

> **[예제 4-16] 50% 알파 블렌딩(4.16_blending_alpha.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 합성에 사용할 알파값
> alpha = 0.5
> 
> # 합성에 사용할 이미지 읽기
> img1 = cv2.imread('./img/wing_wall.jpg')
> img2 = cv2.imread('./img/yate.jpg')
> 
> # 수식을 직접 연산해서 알파 블렌딩 적용
> blended = img1 * alpha + img2 * (1-alpha)
> blended = blended.astype(np.uint8)   # 소수점 제거
> 
> # addWeighted() 함수로 알파 블렌딩 적용
> dst = cv2.addWeighted(img1, alpha, img2, (1-alpha), 0)
> 
> imgs = {'img1 * alpha + img2 * (1-alpha)':blended, 'cv2.addWeighted':dst}
> 
> # 이미지 출력하기
> for i, (k, v) in enumerate(imgs.items()):
>     plt.subplot(1, 2, i + 1)
>     plt.imshow(v[:, :, ::-1])
>     plt.title(k)
>     plt.xticks([])
>     plt.yticks([])
>     
> plt.show()
> ```
> 
> ![Untitled 1](https://user-images.githubusercontent.com/69300448/221179506-2de44aa4-129f-4e9b-87cf-ab8eb0a4a3b6.png)
> 

> **[예제 4-17] 트랙바로 알파 블렌딩(4.17_blending_alpha_trackbar.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> win_name = 'Alpha blending'
> trackbar_name = 'fade'
> 
> # 트랙바 이벤트 핸들러
> def onChange(x):
>     alpha = x/100
>     dst = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)
>     cv2.imshow(win_name, dst)
> 
> # 합성할 이미지 읽기
> img1 = cv2.imread('./img/man_face.jpg')
> img2 = cv2.imread('./img/lion_face.jpg')
> 
> # 이미지 표시 및 트랙바 붙이기
> cv2.imshow(win_name, img1)
> cv2.createTrackbar(trackbar_name, win_name, 0, 100, onChange)
> 
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 2](https://user-images.githubusercontent.com/69300448/221179590-6b153fea-ce00-47d5-b347-c69b204fd5dd.png)
> 

### 4.4.3 비트와이즈 연산

OpenCV는 두 영상의 각 픽셀에 대한 비트와이즈(bitwise, 비트 단위) 연산 기능을 제공한다. 비트와이즈 연산은 두 영상을 합성할 때 특정 영역만 선택하거나 특정 영역만 제외하는 등의 선별적인 연산에 도움이 된다.

- **bitwise_and(img1, img2, mask=None)** : 각 픽셀에 대해 비트와이즈 AND 연산
- **bitwise_or(img1, img2, mask=None)** : 각 픽셀에 대해 비트와이즈 OR 연산
- **bitwise_xor(img1, img2, mask=None)** : 각 픽셀에 대해 비트와이즈 XOR 연산
- **bitwise_not(img1, mask=None)** : 각 픽셀에 대해 비트와이즈 NOT 연산
    - img1, img2 : 연산 대상 영상, 동일한 shape
    - mask : 0이 아닌 픽셀만 연산, 바이너리 이미지
    

> **[예제 4-18] 비트와이즈 연산(4.18_bitwise.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 연산에 사용할 이미지 생성
> img1 = np.zeros((200, 400), dtype=np.uint8)
> img2 = np.zeros((200, 400), dtype=np.uint8)
> img1[:, :200] = 255
> img2[100:200,:] = 255
> 
> # 비트와이즈 연산
> bitAnd = cv2.bitwise_and(img1, img2)
> bitOr = cv2.bitwise_or(img1, img2)
> bitXor = cv2.bitwise_xor(img1, img2)
> bitNot = cv2.bitwise_not(img1)
> 
> # plot으로 결과 출력
> imgs = {'img1':img1, 'img2':img2, 'and':bitAnd,
>        'or':bitOr, 'xor':bitXor, 'not(img1)':bitNot}
> 
> for i, (title, img) in enumerate(imgs.items()):
>     plt.subplot(3, 2, i+1)
>     plt.title(title)
>     plt.imshow(img, 'gray')
>     plt.xticks([])
>     plt.yticks([])
>     
> plt.show()
> ```
> 
> ![Untitled 3](https://user-images.githubusercontent.com/69300448/221179652-1dbe7b1b-b90f-42f4-8bc4-e13c893fa528.png)
> 

비트와이즈 연산으로 영상의 일부분을 원하는 모양으로 떼어낼 수도 있다.

> **[예제 4-19] bitwise_and 연산으로 마스킹하기(4.19_bitwise_masking.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 이미지 로드
> img = cv2.imread('./img/girl.jpg')
> 
> # 마스크 만들기
> mask = np.zeros_like(img)
> cv2.circle(mask, (150, 140), 100, (255, 255, 255), -1)
> 
> # 마스킹
> masked = cv2.bitwise_and(img, mask)
> 
> # plot으로 결과 출력
> imgs = {'original':img, 'mask':mask, 'masked':masked}
> 
> for i, (title, img) in enumerate(imgs.items()):
>     plt.subplot(1, 3, i+1)
>     plt.title(title)
>     plt.imshow(img[:, :, ::-1])
>     plt.xticks([])
>     plt.yticks([])
>     
> plt.show()
> ```
> 
> ![Untitled 4](https://user-images.githubusercontent.com/69300448/221179699-76aa61e8-2fab-4194-9df0-90d0e84795a2.png)
> 

마스킹 부분에서 원본 이미지와 마스크 이미지의 cv2.bitwise_and() 연산으로 원 이외의 부분을 모두 0으로 채워서 원하는 영역만 떼어낼 수 있다.

위 예제에서는 마스크 이미지를 원본과 같은 3채널 배열로 생성했다. 비트와이즈 연산 함수의 세번째 인자인 mask를 이용하면 2차원 배열로도 적용 가능하다.

```python
# 마스크 만들기
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.circle(mask, (150, 140), 100, (255), -1)

# 마스킹
masked = cv2.bitwise_and(img, img, mask=mask)
```

### 4.4.4 차영상

영상에서 영상을 빼기 연산하면 두 영상의 차이, 즉 변화를 알 수 있다.

틀린 그림 찾기 같은 문제 풀이시 손쉽게 답을 찾을 수 있다. 산업현장에서 도면의 차이를 찾거나, PCB 회로의 오류를 찾는데도 사용 가능하고 카메라 촬영 영상에서 실시간으로 움직임이 있는지를 알아내는데도 유용하다.

차영상에서 무턱대고 빼기 연산을 하면 음수가 나올 수 있으므로 절대값을 구해줘야 한다.

- diff = cv2.absdiff(img1, img2)
    - img1, img2 : 입력 영상
    - diff : 두 영상의 차의 절대값 반환

> **[예제 4-20] 차영상으로 도면의 차이 찾아내기(4.20_diff_absolute.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 이미지 읽기
> img1 = cv2.imread('./img/robot_arm1.jpg')
> img2 = cv2.imread('./img/robot_arm2.jpg')
> img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
> img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
> 
> # 두 이미지 절대값 차 연산
> diff = cv2.absdiff(img1_gray, img2_gray)
> 
> # 차영상 극대화하기 위해 스레시홀드 처리 및 컬러로 변환
> _, diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)
> diff_red = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
> diff_red[:, :, 2] = 0
> 
> # 두 번째 이미지에 다른 부분 표시
> spot = cv2.bitwise_xor(img2, diff_red)
> 
> # 결과 출력
> imgs = {'img1':img1, 'img2':img2, 'diff':diff, 'spot':spot}
> 
> for i, (k, v) in enumerate(imgs.items()):
>     plt.subplot(2, 2, i+1)
>     plt.title(k)
>     plt.imshow(v)
>     plt.xticks([])
>     plt.yticks([])
>     
> plt.show()
> ```
> 
> ![Untitled 5](https://user-images.githubusercontent.com/69300448/221179752-1a2d651a-41d6-4fcb-8fe2-790254822fce.png)
> 

### 4.4.5 이미지 합성과 마스킹

두 개의 이미지를 특정 영역끼리 합성하기 위해서 전경이 될 영상과 배경이 될 영상에서 합성하고자 하는 영역만 떼어내는 작업과 그것을 합하는 작업으로 나눌 수 있다. 원하는 영역을 떼어내는 데 꼭 필요한 것은 마스크(mask)인데, 이 작업은 객체 인식과 분리라는 컴퓨터 비전 분야의 한 종류이다.

마스크를 이용하여 전경과 배경을 나누는 것은 **cv2.bitwise_and()** 연산을 이용하면 쉽다.

> **[예제 4-21] 투명 배경 PNG 파일을 이용한 합성(4.21_addition_rgba_mask.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 합성에 사용할 이미지 읽기, 전경 이미지는 4채널 png 파일
> img_fg = cv2.imread('./img/opencv_logo.png', cv2.IMREAD_UNCHANGED)
> img_bg = cv2.imread('./img/girl.jpg')
> 
> # 알파 채널을 이용해서 마스크와 역마스크 생성
> _, mask = cv2.threshold(img_fg[:, :, 3], 1, 255, cv2.THRESH_BINARY)
> mask_inv = cv2.bitwise_not(mask)
> 
> print(img_fg.shape)
> 
> >>> (120, 98, 4)
> 
> # 전경 영상 크기로 배경 영상에서 ROI 잘라내기
> img_fg = cv2.cvtColor(img_fg, cv2.COLOR_BGRA2BGR)
> print(img_fg.shape)
> 
> >>> (120, 98, 3)
> 
> h, w = img_fg.shape[:2]
> roi = img_bg[10:10+h, 10:10+w]
> 
> # 마스크 이용해서 오려내기
> masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
> masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
> 
> # 이미지 합성
> added = masked_fg + masked_bg
> img_bg[10:10+h, 10:10+w] = added
> 
> # 이미지 출력해서 확인하기
> use_window = False
> 
> if use_window:
>     cv2.imshow('mask', mask)
>     cv2.imshow('mask_inv', mask_inv)
>     cv2.imshow('mask_fg', masked_fg)
>     cv2.imshow('masked_bg', masked_bg)
>     cv2.imshow('added', added)
>     cv2.imshow('result', img_bg)
>     cv2.waitKey()
>     cv2.destroyAllWindows()
> else:
>     imgs = {'mask':mask, 'mask_inv':mask_inv, 'masked_fg':masked_fg, 
>             'masked_bg':masked_bg, 'added':added, 'img_bg':img_bg} 
>     for i, (k, v) in enumerate(imgs.items()):
>         plt.subplot(2, 3, i+1)
>         plt.title(k)
>         if len(v.shape) == 2:
>             plt.imshow(v, cmap='gray')
>         else:
>             plt.imshow(v[:,:,::-1])
>         plt.xticks([])
>         plt.yticks([])
>     plt.show()
> ```
> 
> ![Untitled 6](https://user-images.githubusercontent.com/69300448/221179795-1a96aafd-c656-4dee-bc81-383dab657ee8.png)
> 

색상에 따라 영역을 떼어내야 하는 경우도 있을 수 있다. 이때는 색을 가지고 마스크를 만들어야 하는데, HSV로 변환하면 원하는 색상 범위의 것만 골라 낼 수 있다.

OpenCV에서는 특정 범위에 속하는지 여부에 대한 함수를 제공한다.

- dst = cv2.inRange(img, from, to) : 범위에 속하지 않은 픽셀 판단
    - img : 입력 영상
    - from : 범위의 시작 배열
    - to : 범위의 끝 배열
    - dst : img가 from ~ to 에 포함되면 255, 아니면 0을 픽셀 값으로 하는 배열

> **[예제 4-22] HSV 색상으로 마스킹(4.22_hsv_color_mask.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 큐브 이미지를 읽어서 HSV로 변환
> img = cv2.imread('./img/cube.jpg')
> hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
> 
> # 색상별 영역 지정
> blue1 = np.array([90, 50, 50])
> blue2 = np.array([120, 255, 255])
> green1 = np.array([45, 50, 50])
> green2 = np.array([75, 255, 255])
> red1 = np.array([0, 50, 50])
> red2 = np.array([15, 255, 255])
> red3 = np.array([165, 50, 50])
> red4 = np.array([180, 255, 255])
> yellow1 = np.array([20, 50, 50])
> yellow2 = np.array([35, 255, 255])
> 
> # 색상에 따른 마스크 생성
> mask_blue = cv2.inRange(hsv, blue1, blue2)
> mask_green = cv2.inRange(hsv, green1, green2)
> mask_red = cv2.inRange(hsv, red1, red2)
> mask_red2 = cv2.inRange(hsv, red3, red4)
> mask_yellow = cv2.inRange(hsv, yellow1, yellow2)
> 
> # 색상별 마스크로 색상만 추출
> res_blue = cv2.bitwise_and(img, img, mask=mask_blue)
> res_green = cv2.bitwise_and(img, img, mask=mask_green)
> res_red1 = cv2.bitwise_and(img, img, mask=mask_red)
> res_red2 = cv2.bitwise_and(img, img, mask=mask_red2)
> res_red = cv2.bitwise_or(res_red1, res_red2)
> res_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)
> 
> # 이미지 출력하기
> imgs = {'original':img, 'blue':res_blue, 'green':res_green,
>        'red':res_red, 'yellow':res_yellow}
> 
> for i, (k, v) in enumerate(imgs.items()):
>     plt.subplot(2, 3, i+1)
>     plt.title(k)
>     plt.imshow(v[:, :, ::-1])
>     plt.xticks([])
>     plt.yticks([])
>     
> plt.show()
> ```
> 
> ![Untitled 7](https://user-images.githubusercontent.com/69300448/221179844-0ec18f48-074d-49d6-a157-bb249794fd11.png)
> 
> ![Untitled 8](https://user-images.githubusercontent.com/69300448/221179880-75263558-06f0-407f-9a26-17ec978de2eb.png)
> 

**cv2.inRange()** 함수를 호출해서 각 색상 범위별 마스크를 만든다. 함수의 반환 결과는 바이너리 스케일이 되어 **cv2.bitwise_and()** 함수의 mask로 사용하기 적합하다.

위와 같이 색상을 이용한 마스크를 사용하는 것은 크로마 키(chroma key)의 원리와 같다. 초록색 또는 파란색 배경을 두고 찍어서 나중에 원하는 배경과 합성할 때 그 배경을 크로마 키라고 한다. 

크로마 키를 배경으로 한 이미지에서 크로마 키 색상으로 마스크를 만들어서 합성을 해보자.

> **[예제 4-23] 크로마 키 마스킹과 합성(4.23_chromakey.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 크로마키 이미지와 합성할 이미지 불러오기
> img1 = cv2.imread('./img/man_chromakey.jpg')
> img2 = cv2.imread('./img/street.jpg')
> 
> # ROI 선택을 위한 좌표 계산
> h1, w1 = img1.shape[:2]
> h2, w2 = img2.shape[:2]
> x = (w2 - w1) // 2
> y = h2 - h1
> w = x + w1
> h = y + h1
> 
> # 크로마 키 배경 이미지에서 크로마 키가 있을 법한 영역을 10픽셀 정도로 지정하기
> chromakey = img1[:10, :10, :]
> offset = 20
> 
> # 크로마 키 영역 HSV로 변경
> hsv_chroma = cv2.cvtColor(chromakey, cv2.COLOR_BGR2HSV)
> hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
> 
> # 크로마 키 영역의 H 값에서 offset만큼 여유를 두고 범위 지정
> # offset 값은 여러 차례 시도 후 결정
> chroma_h = hsv_chroma[:, :, 0]
> lower = np.array([chroma_h.min()-offset, 100, 100])
> upper  = np.array([chroma_h.max()+offset, 255, 255])
> 
> # 마스크 생성 및 마스킹 후 합성
> mask = cv2.inRange(hsv_img, lower, upper)
> mask_inv = cv2.bitwise_not(mask)
> roi = img2[y:h, x:w]
> fg = cv2.bitwise_and(img1, img1, mask=mask_inv)
> bg = cv2.bitwise_and(roi, roi, mask=mask)
> img2[y:h, x:w] = fg + bg
> 
> print(img1.shape)
> 
> >>> (400, 314, 3)
> 
> # 결과 출력
> plt.title('img1')
> plt.imshow(img1[:,:,::-1])
> plt.xticks([])
> plt.yticks([])
> plt.show()
> 
> plt.title('img2')
> plt.imshow(img2[:,:,::-1])
> plt.xticks([])
> plt.yticks([])
> 
> plt.show()
> ```
> 
> ![Untitled 9](https://user-images.githubusercontent.com/69300448/221179937-87b2af3b-75e9-43fd-bd5a-a573e482507b.png)
> 

이미지 합성에는 대부분 알파 블렌딩 또는 마스킹이 필요하다. 하지만 이런 작업들은 적절한 알파 값 선택과 마스킹을 위한 좌표나 색상값 선택에 많은 노력과 시간이 필요하다. OpenCV는 3 버전에서 새로운 함수를 추가했는데, 알아서 두 영상의 특징을 살려 합성하는 기능이다.

- **dst = cv2.seamlessClone(src, dst, mask, coords, flags[, output])**
    - src : 입력 영상, 일반적으로 전경
    - dst : 대상 영상, 일반적으로 배경
    - mask : 마스크, src에서 합성하고자 하는 영역은 255, 나머지는 0
    - coords : src가 놓여지기 원하는 dst의 좌표(중앙)
    - flags : 합성 방식
        - cv2.NORMAL_CLONE : 입력 원본 유지
        - cv2.MIXED_CLONE : 입력과 대상을 혼합
    - output : 합성 결과
    - dst : 합성 결과

> **[예제 4-24] SeamleassClone으로 합성(4.24_seamlessclone.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # 이미지 불러오기
> img1 = cv2.imread('./img/drawing.jpg')
> img2 = cv2.imread('./img/my_hand.jpg')
> 
> # 마스크 생성, 합성할 이미지 전체 영역을 255로 세팅
> mask = np.full_like(img1, 255)
> 
> # 합성할 좌표 계산(img2의 중앙)
> height, width = img2.shape[:2]
> center = (width//2, height//2)
> 
> # seamlessClone으로 합성
> normal = cv2.seamlessClone(img1, img2, mask, center, cv2.NORMAL_CLONE)
> mixed = cv2.seamlessClone(img1, img2, mask, center, cv2.MIXED_CLONE)
> 
> # 결과 출력
> plt.subplot(1, 2, 1)
> plt.title('normal')
> plt.imshow(normal[:, :, ::-1])
> plt.xticks([])
> plt.yticks([])
> 
> plt.subplot(1, 2, 2)
> plt.title('mixed')
> plt.imshow(mixed[:, :, ::-1])
> plt.xticks([])
> plt.yticks([])
> 
> plt.show()
> ```
> 
> ![Untitled 10](https://user-images.githubusercontent.com/69300448/221179989-23a3a8cf-2b20-41cb-a7e8-4be1f0b0c79f.png)
> 

위 예제에서는 img1의 전체 영역을 255로 채워서 해당 영역 전부가 합성의 대상임을 표시했지만, 가급적이면 합성하려는 영역을 제외하고 0으로 채우는 것이 더 좋은 결과를 보여준다.
