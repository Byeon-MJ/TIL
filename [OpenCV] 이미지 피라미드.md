# 6.4 이미지 피라미드

**이미지 피라미드(Image pyramides)** 는 영상의 크기를 단계적으로 축소 또는 확대해서 피라미드처럼 쌓아놓는 것을 말한다.

영상을 분석할 때 작은 이미지로 빠르게 확인하고 다음 단계 크기의 영상으로 분석하는 식으로 정확도를 높이는 것이 효율적이고, 영상의 크기에 따라 분석하는 내용이 다를 수 있다.

### 6.4.1 가우시안 피라미드

가우시안 필터를 적용한 후에 이미지 피라미드를 구현하는 것을 말한다.

- **dst = cv2.pyrDown(src [, dst, dstsize, borderType])**
- **dst = cv2.pyrUp(src [, dst, dstsize, borderType])**
    - src : 입력 영상, Numpy 배열
    - dst : 결과 영상
    - dstsize : 결과 영상 크기
    - borderType : 외곽 보정 방식

**cv2.pyrDown()** 함수는 가우시안 필터를 적용하고 모든 짝수 행과 열을 삭제하여 입력 영상의 1/4 크기로 축소한다. dstsize 인자는 원하는 결과 크기를 지정할 수 있지만, 아무 크기나 지정할 수는 없어서 사용하기 까다롭다.

**cv2.pyrup()** 함수는 0으로 채워진 짝수 행과 열을 새롭게 삽입하고 가우시안 필터로 컨볼루션을 수행해서 주변 픽셀과 비슷하게 만드는 방법으로 4배 확대한다.

> **[예제 6-18] 가우시안 이미지 피라미드(6.18_pyramid_gaussian.ipynb)**
> 
> 
> ```python
> import cv2
> 
> img = cv2.imread('./img/girl.jpg')
> 
> # Gaussian image pyramid reduction
> smaller = cv2.pyrDown(img)  # img x 1/4
> 
> # Gaussian image pyramid expansion
> bigger = cv2.pyrUp(img)  # img x 4
> 
> # output result
> cv2.imshow('img', img)
> cv2.imshow('pyrDown', smaller)
> cv2.imshow('pyrUp', bigger)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](6%204%20%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20%E1%84%91%E1%85%B5%E1%84%85%E1%85%A1%E1%84%86%E1%85%B5%E1%84%83%E1%85%B3%2060b6eb98dd4c4f16aa006ee6e3fe9512/Untitled.png)
> 

### 6.4.2 라플라시안 피라미드

cv2.pyUp() 함수는 4배로 확대할 때 없던 행과 열을 생성해서 가우시안 필터를 적용하기때문에 원래의 영상만큼 완벽하지 않다. 

원본 영상에서 cv2.pyUp()으로 확대한 영상을 빼면 원본과 확대본의 차이가 되는데, 이것을 보관해 두었다가 확대 영상에 더하면 원본을 완벽히 복원할 수 있다. 원본과 cv2.pyUp() 함수를 적용한 영상의 차이를 단계별로 모아두는 것을 **라플라시안 피라미드** 라고 한다.

> **[예제 6-19] 라플라시안 피라미드로 영상 복원(6.19_pyramid_laplacian.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/taekwonv1.jpg')
> 
> # Reduce the original video using a Gaussian pyramid
> smaller = cv2.pyrDown(img)
> 
> # Enlarge the reduced video using a Gaussian pyramid
> bigger = cv2.pyrUp(smaller)
> 
> # Subtract the enlarged video from the original
> laplacian = cv2.subtract(img, bigger)
> 
> # Restore by adding a Laplacian image to an enlarged image
> restored = bigger + laplacian
> 
> # output result
> merged = np.hstack((img, laplacian, bigger, restored))
> cv2.imshow('Laplacian Pyramid', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](6%204%20%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%20%E1%84%91%E1%85%B5%E1%84%85%E1%85%A1%E1%84%86%E1%85%B5%E1%84%83%E1%85%B3%2060b6eb98dd4c4f16aa006ee6e3fe9512/Untitled%201.png)
> 

라플라시안 피라미드는 영상의 복원 목적 뿐만 아니라 경계 검출에도 활용할 수 있다.
