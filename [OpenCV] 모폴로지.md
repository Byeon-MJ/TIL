# 6.3 모폴로지

**모폴로지(morphology)** 는 영상 분야에서 노이즈 제거, 구멍 메꾸기, 연결되지 않은 경계 이어붙이기 등 형태학적 관점에서의 영상 연산을 말한다.

주로 형태를 다루는 연산이므로 바이너리 이미지를 대상으로 한다. 대표적인 연산은 침식과 팽창이며, 이 둘을 결합한 열림과 닫힘 연산 등이 있다.

### 6.3.1 침식 연산

**침식(erosion)** 은 원래 있던 객체의 영역을 깍아내는 연산이다. 연산을 위해서는 **구조화 요소(structuring element)** 라는 0과 1로 채워진 커널이 필요한데, 1이 채워진 모양에 따라 사각형, 타원형, 십자형 등을 사용할 수 있다.

침식 연산은 구조화 요소 커널을 입력 영상에 적용해서 1로 채워진 영역을 온전히 올려 놓을 수 없으면 해당 픽셀을 0으로 변경한다.

OpenCV는 구조화 요소 커널 생성을 위한 함수로 **cv2.getStructuringElement()** 를, 침식 연산을 위한 함수로 **cv2.erode()** 를 제공한다.

- **cv2.getStructuringElement(shape, ksize[, anchor])**
    - shape : 구조화 요소 커널의 모양 결정
        - cv2.MORPH_RECT : 사각형
        - cv2.MORPH_ELLIPSE : 타원형
        - cv2.MORPH_CROSS : 십자형
    - ksize : 커널 크기
    - anchor : 구조화 요소의 기준점, cv2.MORPH_CROSS에만 의미 있고 기본 값은 중심점(-1, -1)
- **dst = cv2.erode(src, kernel [, anchor, iterations, borderType, borderValue])**
    - src : 입력 영상, Numpy 객체, 바이너리 영상(검은색 : 배경, 흰색 : 전경)
    - kernel : 구조화 요소 커널 객체
    - anchor : cv2.getStructuringElement()와 동일
    - iterations : 침식 연산 적용 반복 횟수
    - borderType : 외곽 영역 보정 방법 설정 플래그
    - borderValue : 외곽 영역 보정값

침식 연산은 큰 물체는 주변을 깎아서 작게 만들지만 작은 객체는 아예 사라지게 만들 수 있으므로 아주 작은 노이즈를 제거하거나, 따로 떨어진 물체인데 겹쳐져서 하나의 물체로 보일 때 서로를 떼어내는 데도 효과적이다.

> **[예제 6-13] 침식 연산(6.13_morph_erode.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/morph_dot.png')
> 
> # Structuring element kernel, Create rectangle(3 x 3)
> k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
> 
> # Apply erosion operation
> erosion = cv2.erode(img, k)
> 
> # Output result
> merged = np.hstack((img, erosion))
> cv2.imshow('Erode', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](https://user-images.githubusercontent.com/69300448/225782721-c126254b-ef67-4cc3-8d30-9c58073e763a.png)
> 

결과를 보면 글씨가 전반적으로 홀쭉해지긴 했지만 작은 흰 점들로 구성된 노이즈가 사라진 것을 알 수 있다.

### 6.3.2 팽창 연산

**팽창(dilatation)** 은 침식과는 반대로 영상 속 사물의 주변을 덧붙여서 영역을 더 확장하는 연산이다. 구조화 요소 커널을 입력 영상에 적용해서 1로 채워진 영역이 온전히 덮이지 않으면 1로 채워 넣는다.

- **dst = cv2.dilate(src, kernel[, dst, anchor, iterations, borderType, borderValue]) :**
    - 모든 인자는 cv2.erode() 함수와 동일

> **[예제 6-14] 팽창 연산(6.14_morph_dilate.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/morph_hole.png')
> 
> # Structuring element kernel, Create rectangle(3 x 3)
> k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
> 
> # Apply dilatation operation
> dst = cv2.dilate(img, k)
> 
> # Output result
> merged = np.hstack((img, dst))
> cv2.imshow('Dilation', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 1](https://user-images.githubusercontent.com/69300448/225782739-5f2f7434-ebdc-4d44-9ce9-7e79645463dc.png)
> 

글씨가 조금 뚱뚱해지긴 했지만, 글씨 안의 점 노이즈가 사라졌다.

### 6.3.3 열림과 닫힘, 그 밖의 모폴로지 연산

[Morphological Transformations — gramman 0.1 documentation](https://opencv-python.readthedocs.io/en/latest/doc/12.imageMorphological/imageMorphological.html)

[모폴로지 연산의 유형
- MATLAB & Simulink
- MathWorks 한국](https://kr.mathworks.com/help/images/morphological-dilation-and-erosion.html)

침식과 팽창 연산은 밝은 부분이나 어두운 부분의 점 노이즈를 없애는 데 효과적이다. 하지만 원래의 모양이 홀쭉해지거나 뚱뚱해지는 변형이 일어난다.

침식과 팽창 연산을 조합하면 원래의 모양을 유지하면서 노이즈만 제거할 수 있다. 

**모폴로지 연산**

- 열림(Opening) 연산
    - 침식 → 팽창 연산 적용
    - 주변보다 밝은 노이즈 제거에 효과적
    - 맞닿아 있는 것으로 보이는 독립된 개체를 분리하거나 돌출된 픽셀을 제거
- 닫힘(Closing) 연산
    - 팽창 → 침식 연산 적용
    - 주변보다 어두운 노이즈 제거에 효과적
    - 끊어져 보이는 개체를 연결하거나 구멍을 메우는 데 사용
- 그래디언트(Gradient) 연산
    - 팽창 - 침식
    - 팽창한 결과에서 침식한 결과를 빼서 경계만 얻어낸 것
    - 경계 검출과 비슷한 결과를 얻을 수 있다.
- 탑햇(top hat) 연산
    - 원본 - 열림
    - 원본에서 열림 연산 결과를 빼면 밝기 값이 크게 튀는 영역을 강조
- 블랙햇(black hat) 연산
    - 닫힘 - 원본
    - 닫힘 연산 결과에서 원본을 빼면 어두운 부분을 강조
    

OpenCV는 열림과 닫힘 연산 등의 모폴로지 연산을 위해 함수를 제공한다.

- dst = cv2.morphologyEx(src, op, kernel [, dst, anchor, iteration, borderType, borderValue])
    - src : 입력 영상, Numpy 배열
    - op : 모폴로지 연산 종류 지정
        - cv2.MORPH_OPEN : 열림 연산
        - cv2.MORPH_CLOSE : 닫힘 연산
        - cv2.MORPH_GRADIENT : 그래디언트 연산
        - cv2.MORPH_TOPHAT : 탑햇 연산
        - cv2.MORPH_BLACKHAT : 블랙햇 연산
    - kernel : 구조화 요소 커널
    - dst : 결과 영상
    - anchor : 커널의 기준점
    - iteration : 연산 반복 횟수
    - borderType : 외곽 보정 방식
    - borderValue : 외곽 보정 값

> **[예제 6-15] 열림과 닫힘 연산으로 노이즈 제거(6.15_morph_open_close.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img1 = cv2.imread('./img/morph_dot.png', cv2.IMREAD_GRAYSCALE)
> img2 = cv2.imread('./img/morph_hole.png', cv2.IMREAD_GRAYSCALE)
> 
> # Structuring element kernel, Create rectangle(5 x 5)
> k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
> 
> # Apply opening operation
> opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, k)
> 
> # Apply closing operation
> closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, k)
> 
> # Output result
> merged1 = np.hstack((img1, opening))
> merged2 = np.hstack((img2, closing))
> merged = np.vstack((merged1, merged2))
> cv2.imshow('opening, closing', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 2](https://user-images.githubusercontent.com/69300448/225782758-8ad88f37-f9b5-4818-b859-fabf5871806c.png)
> 

팽창, 침식과 달리 원본 영상이 뚱뚱해지거나 홀쭉해지지 않고 원래의 크기를 그대로 유지한 채 노이즈를 제거한다.

> **[예제 6-16] 모폴로지 그래디언트(6.16_morph_gradient.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/morphological.png')
> 
> # Structuring element kernel, Create rectangle(3 x 3)
> k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
> 
> # Apply gradient operation
> gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)
> 
> # output result
> merged = np.hstack((img, gradient))
> cv2.imshow('gradient', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 3](https://user-images.githubusercontent.com/69300448/225782778-b1cdafde-5e39-46b9-835a-f335cbb3a4ca.png)
> 

> **[예제 6-17] 모폴로지 탑햇, 블랙햇 연산(6.17_morph_hat.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/moon_gray.jpg')
> 
> # structuring element kernel, create rectangle(5 x 5)
> k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
> 
> # apply top-hat operation
> tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k)
> 
> # apply black-hat operation
> blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k)
> 
> # output result
> merged = np.hstack((img, tophat, blackhat))
> cv2.imshow('tophat blackhat', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 4](https://user-images.githubusercontent.com/69300448/225782785-018cea1e-950d-4673-82ea-f6ed9731208f.png)
>
