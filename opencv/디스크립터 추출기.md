# 8.3 디스크립터 추출기

### 8.3.1 특징 디스크립터와 추출기

키 포인트는 영상의 특징이 있는 픽셀의 좌표와 그 주변 픽셀과의 관계에 대한 정보를 가진다.

가장 대표적인 것이 size와 angle 속성으로, 코너 특징인 경우 엣지의 경사도 규모와 방향을 나타낸다.

**특징 디스크립터는(Feature descriptor)** 는 키 포인트 주변 픽셀을 일정한 크기의 블록으로 나누어 각 블록에 속한 픽셀의 그래디언트 히스토그램을 계산한 것으로, 키 포인트 주변의 밝기, 색상, 방향, 크기 등의 정보를 표현한 것이다. 

특징을 나타내는 값을 매칭에 사용하기 위해서는 회전, 크기, 방향 등에 영향이 없어야 하는데, 이를 위해 특징 디스크립터가 필요하다.

특징 디스크립터를 추출하는 알고리즘에 따라 내용, 모양 그리고 크기가 다를 수 있지만 일반적으로는 키 포인트에 적용하는 주변 블록의 크기에 8방향의 경사도를 표현하는 형태인 경우가 많다.

OpenCV는 특징 디스크립터를 추출하기 위한 방법으로 통일된 인터페이스를 제공하기 위해 특징 검출기와 같은 cv2.Feature2D 클래스를 상속받아 구현

- **keypoints, descriptors = detector.compute(image, keypoints[, descriptors])** : 키 포인트를 전달하면 특징 디스크립터를 계산해서 반환
- **keypoints, descriptors = detector.detectAndCompute(image, mask[, descriptors, useProvidedKeypoints])** : 키 포인트 검출과 특징 디스크립터 계산을 한번에 수행
    - image : 입력 영상
    - keypoints : 디스크립터 계산을 위해 사용할 키 포인트
    - descriptors : 계산된 디스크립터
    - mask : 키 포인트 검출에 사용할 마스크
    - useProvidedKeypoints : True인 경우 키 포인트 검출을 수행하지 않음(사용 안 함)

cv2.Feature2D를 상속받은 몇몇 특징 검출기는 detect() 함수만 구현되어 있고 compute()와 detectAndCompute() 함수는 구현되어 있지 않은 경우도 있고 그 반대의 경우도 있다.

### 8.3.2 SIFT**(Scale-Invariant Feature Transform)**

이미지 피라미드를 이용해서 크기 변화에 따른 특징 검출의 문제를 해결한 알고리즘

특허권이 있어 상업적 사용에는 제약이 있고 OpenCV는 contrib 모듈에만 포함

- **detector = cv2.xfeatures2d.SIFT_create([, nfeatures[, nOctaveLayers[, contrastThreshold[, edgeThreshold[, sigma]]]]])**
    - nfeatures : 검출 최대 특징 수
    - nOctaveLayers : 이미지 피라미드에 사용할 계층 수
    - contrastThreshold : 필터링할 빈약한 특징 스레시홀드
    - edgeThreshold : 필터링할 엣지 스레시홀드
    - sigma : 이미지 피라미드 0 계층에서 사용할 가우시안 필터의 시그마 값

> **[예제 8-10] SIFT로 키 포인트 및 디스크립터 추출(8.10_desc_sift.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/house.jpg')
> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> 
> # SIFT 추출기 생성
> sift = cv2.xfeatures2d.SIFT_create()
> 
> # 키 포인트 검출과 디스크립터 계산
> keypoints, descriptor = sift.detectAndCompute(gray, None)
> 
> print('keypoint:', len(keypoints), 'descriptor:', descriptor.shape)
> print(descriptor)
> 
> >>> keypoint: 413 descriptor: (413, 128)
> [[  1.   1.   1. ...   0.   0.   1.]
>  [  8.  24.   0. ...   1.   0.   4.]
>  [  0.   0.   0. ...   0.   0.   2.]
>  ...
>  [  1.   8.  71. ...  73. 127.   3.]
>  [ 35.   2.   7. ...   0.   0.   9.]
>  [ 36.  34.   3. ...   0.   0.   1.]]
> 
> # 키 포인트 그리기
> img_draw = cv2.drawKeypoints(img, keypoints, None, 
>                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
> 
> # 결과 출력
> cv2.imshow('SIFT', img_draw)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](https://user-images.githubusercontent.com/69300448/235801345-a4fffa25-82d0-46e1-bc4f-da30896ca09e.png)
> 

SIFT를 이용한 키 포인트와 디스크립터를 계산해서 결과 화면에 표시하고 디스크립터의 크기와 일부 데이터를 출력한다. 413개의 키 포인트가 검출되었고 키 포인트 1개당 128개의 특징 벡터값을 사용한다.

### 8.3.3 SURF**(Speeded Up Robust Features)**

이미지 피라미드를 사용하여 특징 검출을 하는 SIFT의 속도가 느리다는 단점을 개선, 이미지 피라미드 대신 커널 크기를 바꾸는 방식으로 성능을 개선한 알고리즘

특허권이 있어 상업적 이용에 제한이 있고 OpenCV contrib 모듈에도 포함되어 있지 않음

기능을 사용하기 위해서는 소스 코드를 직접 빌드하면서 OPENCV_ENABLE_NONFREE=ON 옵션을 지정

- **detector = cv2.xfeatures2d.SURF_create([hessianThreshold, nOctaves, nOctaveLayers, extended, upright])**
    - hessianThreshold : 특징 추출 경계값(100)
    - nOctaves : 이미지 피라미드 계층수(3)
    - extended : 디스크럽터 생성 플래그(False), True : 128개, False : 64개
    - upright : 방향 개선 플래그(False), True : 방향 무시, False : 방향 적용

> **[예제 8-11] SURF로 키 포인트 및 디스크립터 추출(8.11_desc_surf.ipynb)**
> 
> 
> → SURF는 특허와 관련된 문제로 OpenCV 3.4.2 버전 이상에서는 지원하지 않는다. 
> 아래 코드는 **`opencv-contrib-python 3.4.2`** 버전을 사용 또는 **`OpenCV cMake build, OPENCV_ENABLE_NONFREE=ON`** 설정을 통해 실행 가능
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/house.jpg')
> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> 
> # SURF 추출기 생성(경계:1000, 피라미드:3, 서술자 확장:True, 방향 적용:True)
> surf = cv2.xfeatures2d_SURF.create(1000, 3, extended=True, upright=True)
> 
> # 키 포인트 검출 및 서술자 계산
> keypoints, desc = surf.detectAndCompute(gray, None)
> print(desc.shape, desc)
> 
> # 키 포인트 이미지에 그리기
> img_draw = cv2.drawKeypoints(img, keypoints, None,
>                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
> 
> cv2.imshow('SURF', img_draw)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 

### 8.3.4 ORB**(Oriented and Rotated BRIEF)**

특징 검출을 지원하지 않는 디스크립터 추출기인 **BRIEF(Binary Robust Independent Elementary Features)** 에 방향과 회전을 고려하도록 개선한 알고리즘

특징 검출 알고리즘으로 FAST를 채용, 회전과 방향에 영향을 받지 않으면서 속도가 빠름

특허 때문에 사용에 제약이 많은 SIFT와 SURF의 좋은 대안으로 사용됨

- **detector = cv2.ORB_create([nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold])**
    - nfeatures = 500 : 검출할 최대 특징 수
    - scaleFactor = 1.2 : 이미지 피라미드 비율
    - nlevels = 8 : 이미지 피라미드 계층수
    - edgeThreshold = 31 : 검색에서 제외할 테두리 크기, patchSize와 맞출 것
    - firstLevel = 0 : 최초 이미지 피라미드 계층 단계
    - WTA_K = 2 : 임의 좌표 생성 수
    - scoreType : 키 포인트 검출에 사용할 방식
        - cv2.ORB_HARRIS_SCORE : 해리스 코너 검출(기본값)
        - cv2.ORB_FAST_SCORE : FAST 코너 검출
    - patchSize = 31 : 디스크립터의 패치 크기
    - faaastThreshold = 20 : FAST에 사용할 임계값

> **[예제 8-12] ORB로 키 포인트 및 디스크립터 추출(8.12_desc_orb.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/house.jpg')
> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> 
> # ORB 추출기 생성
> orb = cv2.ORB_create()
> 
> # 키 포인트 검출과 디스크립터 계산
> keypoints, descriptor = orb.detectAndCompute(img, None)
> 
> # 키 포인트 그리기
> img_draw = cv2.drawKeypoints(img, keypoints, None,
>                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
> 
> # 결과 출력
> cv2.imshow('ORB', img_draw)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled 1](https://user-images.githubusercontent.com/69300448/235801391-a2516a20-2d6d-48ac-80ad-647757a0aea5.png)
>
