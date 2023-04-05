# 8.2 영상의 특징과 키 포인트

앞서 다룬 특징 추출과 매칭 방법은 영상 전체를 전역적으로 반영하는 방법이다. 전역적 매칭은 비교하려는 두 영상의 내용이 거의 대부분 비슷해야 하고, 다른 물체에 가려지거나 회전이나 방향, 크기 변화가 있으면 효과가 없다.

그래서 여러 개의 지역적 특징을 표현할 수 있는 방법이 필요하다.

### 8.2.1 코너 특징 검출

사람은 영상 속 내용을 판단할 때 주로 픽셀의 변화가 심한 곳에 중점적으로 관심을 두게 된다. 그 중에서도 엣지와 엣지가 만나는 코너(corner)에 가장 큰 관심을 둔다. 코너는 영상의 특징을 아주 잘 표현하는 요소이기 때문이다.

코너를 검출하기 위한 방법으로는 **해리스 코너 검출(Harris corner detection)** 이 원조격이다. 해리스 코너 검출은 소벨(Sobel) 미분으로 엣지를 검출하면서 엣지의 경사도 변화량을 측정하여 변화량이 X축과 Y축 모든 방향으로 크게 변화하는 것을 코너로 판단한다.

- dst = cv2.cornerHarris(src, blockSize, ksize, k[, dst, borderType])
    - src : 입력 영상, 그레이 스케일
    - blockSize : 이웃 픽셀 범위
    - ksize : 소벨 미분 커널 크기
    - k : 코너 검출 상수, 경험적 상수(0.04 ~ 0.06)
    - dst : 코너 검출 결과
        - src와 같은 크기의 1채널 배열, 변화량의 값, 지역 최대값이 코너점을 의미
    - borderType : 외곽 영역 보정 형식

> **[예제 8-4] 해리스 코너 검출(8.4_corner_harris.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/house.jpg')
> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> 
> # Harris corner detection
> corner = cv2.cornerHarris(gray, 2, 3, 0.04)
> # Get coordinates with a maximum difference of 10% in the result
> coord = np.where(corner > 0.1 * corner.max())
> coord = np.stack((coord[1], coord[0]), axis=-1)
> 
> # Draw circle at the corner coordinates
> for x, y in coord:
>     cv2.circle(img, (x, y), 5, (0, 0, 255), 1, cv2.LINE_AA)
> 
> # Normalize the change amount to 0~255 to represent image
> corner_norm = cv2.normalize(corner, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
> 
> # Output to the screen
> corner_norm = cv2.cvtColor(corner_norm, cv2.COLOR_GRAY2BGR)
> merged = np.hstack((corner_norm, img))
> cv2.imshow('Harris Corner', merged)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](8%202%20%E1%84%8B%E1%85%A7%E1%86%BC%E1%84%89%E1%85%A1%E1%86%BC%E1%84%8B%E1%85%B4%20%E1%84%90%E1%85%B3%E1%86%A8%E1%84%8C%E1%85%B5%E1%86%BC%E1%84%80%E1%85%AA%20%E1%84%8F%E1%85%B5%20%E1%84%91%E1%85%A9%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%90%E1%85%B3%20f338a659949443708d5e97803471b424/Untitled.png)
> 

cv2.cornerHarris() 함수의 결과는 입력 영상과 같은 크기의 1차원 배열로 지역 최대값이 코너를 의미한다.

시(Shi)와 토마시(Tomasi)는 논문을 통해 해리스 코너 검출을 개선한 알고리즘을 발표했다. 이 방법으로 검출한 코너는 객체 추적에 좋은 특징이 된다고 해서, OpenCV에서 **cv2.goodFeaturesToTrack()** 이라는 이름의 함수로 제공한다.

- corners = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance[, corners, mask, blockSize, useHarrisDetector, k])
    - img : 입력 영상
    - maxCorners : 얻고 싶은 코너 개수, 강한 것 순
    - qualityLevel : 코너로 판단할 스레시홀드 값
    - minDistance : 코너 간 최소 거리
    - mask : 검출에 제외할 마스크
    - blockSize=3 : 코너 주변 영역의 크기
    - useHarrisDetector=False : 코너 검출 방법 선택
        - True = 해리스 코너 검출 방법, False = 시와 토마시 검출 방법
    - k : 해리스 코너 검출에 사용할 k 계수
    - corners : 코너 검출 좌표 결과, N x 1 x 2 크기의 배열, 실수값이므로 정수로 변환 필요

> **[예제 8-5] 시-토마시 코너 검출(8.5_corner_goodFeature.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/house.jpg')
> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> 
> # 시-토마스의 코너 검출 메서드
> corners = cv2.goodFeaturesToTrack(gray, 80, 0.01, 10)
> # 실수 좌표를 정수 좌표로 변환
> corners = np.int32(corners)
> 
> # 좌표에 동그라미 표시
> for corner in corners:
>     x, y = corner[0]
>     cv2.circle(img, (x, y), 5, (0, 0, 255), 1, cv2.LINE_AA)
> 
> cv2.imshow('Corners', img)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](8%202%20%E1%84%8B%E1%85%A7%E1%86%BC%E1%84%89%E1%85%A1%E1%86%BC%E1%84%8B%E1%85%B4%20%E1%84%90%E1%85%B3%E1%86%A8%E1%84%8C%E1%85%B5%E1%86%BC%E1%84%80%E1%85%AA%20%E1%84%8F%E1%85%B5%20%E1%84%91%E1%85%A9%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%90%E1%85%B3%20f338a659949443708d5e97803471b424/Untitled%201.png)
> 

### 8.2.2 키 포인트와 특징 검출기

영상에서 특징점을 찾아내는 알고리즘은 무척 다양하다. 각각의 특징점은 픽셀의 좌표 이외에도 표현할 수 있는 정보가 많다.

OpenCV는 여러 특징점 검출 알고리즘 중 어떤 것을 사용하든 간에 동일한 코드로 특징점을 검출할 수 있게 하기 위해서 각 알고리즘 구현 클래스가 추상 클래스를 상속받는 방법으로 인터페이스를 통일했다.

모든 특징점 검출기를 **cv2.Feature2D** 클래스를 상속받아 구현했고, 이것으로부터 추출된 특징점은 **cv2.KeyPoint** 라는 객체에 담아 표현한다. cv2.Feature2D를 상속받아 구현된 특징 검출기는 아래 공식문서 URL에서 확인 가능하다.

[OpenCV: Feature Detection and Description](https://docs.opencv.org/3.4.1/d5/d51/group__features2d__main.html)

**cv2.Feature2D** 를 상속받은 특징 검출기는 **detect()** 함수를 구현하고 있고, 이 함수는 특징점의 좌표와 추가 정보를 담은 **cv2.KeyPoint** 객체를 리스트에 담아 반환한다.

- **keypoints = detector.detect(img [, mask])** : 키 포인트 검출 함수
    - img : 입력 영상, 바이너리 스케일
    - mask : 검출 제외 마스크
    - keypoints : 특징점 정보를 담는 객체
- **KeyPoint** : 특징점 정보를 담는 객체, pt 속성은 항상 값을 갖지만 나머지 속성은 검출기에 따라 채워지지 않는 경우도 존재한다.
    - pt : 키 포인트(x, y) 좌표, float 타입으로 정수로 변환 필요
    - size : 의미 있는 키 포인트 이웃의 반지름
    - angle : 특징점 방향(시계방향, -1 = 의미 없음)
    - response : 특징점 반응 강도(추출기에 따라 다름)
    - octave : 발견된 이미지 피라미드 계층
    - class_id : 키 포인트가 속한 객체 ID
- outImg = cv2.drawKeypoints(img, keypoints, outImg[, color[, flags]]) : 키 포인트 영상 표시 함수
    - img : 입력 이미지
    - keypoints : 표시할 키 포인트 리스트
    - outImg : 키 포인트가 그려진 결과 이미지
    - color : 표시할 색상(기본값 : 랜덤)
    - flags : 표시 방법 선택 플래그
        - cv2.DRAW_MATCHES_FLAGS_DEFAULT : 좌표 중심에 동그라미만 그림(기본값)
        - cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS : 동그라미의 크기를 size와 angle을 반영해서 그림

### 8.2.3 GFTTDetector

**GFTTDectector** 는 앞서 살펴본 cv2.goodFeaturesToTrack() 함수로 구현된 특징 검출기이다. 

검출기 생성 방법만 다르고 사용하는 함수는 cv2.Feature2D의 detect() 함수와 같다.

- **detector = cv2.GFTTDetector_create([, maxCorners[, qualityLevel, minDistance, blockSize, useHarrisDetector, k])**
    - 인자의 모든 내용은 **cv2.goodFeaturesToTrack()** 과 동일

> **[예제 8-6] GFTTDetector로 키 포인트 검출(8.6_kpt_gftt.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/house.jpg')
> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> 
> # Good feature to track 검출기 생성
> gftt = cv2.GFTTDetector_create()
> 
> # 키 포인트 검출
> keypoints = gftt.detect(gray, None)
> 
> # 키 포인트 그리기
> img_draw = cv2.drawKeypoints(img, keypoints, None)
> 
> # 결과 출력
> cv2.imshow('GFTTDectector', img_draw)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](8%202%20%E1%84%8B%E1%85%A7%E1%86%BC%E1%84%89%E1%85%A1%E1%86%BC%E1%84%8B%E1%85%B4%20%E1%84%90%E1%85%B3%E1%86%A8%E1%84%8C%E1%85%B5%E1%86%BC%E1%84%80%E1%85%AA%20%E1%84%8F%E1%85%B5%20%E1%84%91%E1%85%A9%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%90%E1%85%B3%20f338a659949443708d5e97803471b424/Untitled%202.png)
> 

### 8.2.4 FAST

**FAST(Feature from Accelerated Segment Test)** 는 속도를 개선한 알고리즘이다. 2006년 **에드워드 로스텐(Edward Rosten)** 과 **톰 드러먼드(Tom Drummond)** 의 논문에 소개된 알고리즘이다.

코너를 검출할 때 미분 연산으로 엣지 검출을 하지 않고 픽셀을 중심으로 특정 개수의 픽셀로 원을 그려서 그 안의 픽셀들이 중심 픽셀값보다 임계값 이상 밝거나 어두운 것이 특정 개수 이상 연속되면 코너로 판단한다.

- **detector = cv2.FastFeatureDetector_create(threshold[, nonmaxSuppression, type])**
    - threshold = 10 : 코너 판단 임계값
    - nonmaxSuppression = True : 최대 점수가 아닌 코너 억제
    - type : 엣지 검출 패턴
        - cv2.FastFeatureDetector_TYPE_9_16 : 16개 중 9개 연속(기본값)
        - cv2.FastFeatureDetector_TYPE_7_12 : 12개 중 7개 연속
        - cv2.FastFeatureDetector_TYPE_5_8 : 8개 중 5개 연속

> **[예제 8-7] FAST로 키 포인트 검출(8.7_kpt_fast.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/house.jpg')
> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> 
> # FAST 특징 검출기 생성
> fast = cv2.FastFeatureDetector_create(50)
> 
> # 키 포인트 검출
> keypoints = fast.detect(gray, None)
> 
> # 키 포인트 그리기
> img = cv2.drawKeypoints(img, keypoints, None)
> 
> # 결과 출력
> cv2.imshow('FAST', img)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](8%202%20%E1%84%8B%E1%85%A7%E1%86%BC%E1%84%89%E1%85%A1%E1%86%BC%E1%84%8B%E1%85%B4%20%E1%84%90%E1%85%B3%E1%86%A8%E1%84%8C%E1%85%B5%E1%86%BC%E1%84%80%E1%85%AA%20%E1%84%8F%E1%85%B5%20%E1%84%91%E1%85%A9%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%90%E1%85%B3%20f338a659949443708d5e97803471b424/Untitled%203.png)
> 

### 8.2.5 SimpleBlobDetector

**BLOB(Binary Large Object)** 는 바이너리 스케일 이미지의 연결된 픽셀 그룹을 말하는 것이다.

자잘한 객체는 노이즈로 판단하고 특정 크기 이상의 큰 객체에만 관심을 두는 방법이다.

코너를 이용한 특징 검출과는 방식이 다르지만 영상의 특징을 표현하기 좋은 또 하나의 방법이다.

- **detector = cv2.SimpleBlobDetector_create( [parameters] )** : BLOB 검출기 생성자
    - parameters : BLOB 검출 필터 인자 객체
- **cv2.SimpleBlobDetector_Params()**
    - minThreshold, maxThreshold, thresholdStep : BLOB를 생성하기 위한 경계값
    (minThreshold에서 maxThreshold를 넘지 않을 때까지 thresholdStep 만큼 증가)
    - minRepeatability : BLOB에 참여하기 위한 연속된 경계값의 개수
    - minDistBetweenBlobs : 두 BLOB를 하나의 BLOB로 간주한 거리
    - filterByArea : 면적 필터 옵션
    - minArea, maxArea : min~max 범위의 면적만 BLOB로 검출
    - filterByCircularity : 원형 비율 필터 옵션
    - minCircularity, maxCircularity : min~max 범위의 원형 비율만 BLOB로 검출
    - filterByColor : 밝기를 이용한 필터 옵션
    - blobColor : 0 = 검은색 BLOB 검출, 255 = 흰색 BLOB 검출
    - filterByConvexity : 볼록 비율 필터 옵션
    - minConvexity, maxConvexity : min~max 범위의 볼록 비율만 BLOB로 검출
    - filterByInertia : 관성 비율 필터 옵션
    - minInertiaRatio, maxInertiaRatio : min~max 범위의 관성 비율만 BLOB로 검출

> **[예제 8-8] SimpleBlobDetector 검출기(8.8_kpt_blob.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/house.jpg')
> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> 
> # SimpleBlobDetector 생성
> detector = cv2.SimpleBlobDetector_create()
> 
> # 키 포인트 검출
> keypoints = detector.detect(gray)
> 
> # 키 포인트를 빨간색으로 표시
> img = cv2.drawKeypoints(img, keypoints, None, (0, 0, 255), 
>                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
> 
> # 결과 출력
> cv2.imshow('Blob', img)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](8%202%20%E1%84%8B%E1%85%A7%E1%86%BC%E1%84%89%E1%85%A1%E1%86%BC%E1%84%8B%E1%85%B4%20%E1%84%90%E1%85%B3%E1%86%A8%E1%84%8C%E1%85%B5%E1%86%BC%E1%84%80%E1%85%AA%20%E1%84%8F%E1%85%B5%20%E1%84%91%E1%85%A9%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%90%E1%85%B3%20f338a659949443708d5e97803471b424/Untitled%204.png)
> 

> **[예제 8-9] 필터 옵션으로 생성한 SimpleBlobDetector 검출기(8.9_kpt_blob_param.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> img = cv2.imread('./img/house.jpg')
> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> 
> # BLOB 검출 필터 파라미터 생성
> params = cv2.SimpleBlobDetector_Params()
> 
> # 경계값 조정
> params.minThreshold = 10
> params.maxThreshold = 240
> params.thresholdStep = 5
> 
> # 면적 필터를 켜고 최소값 지정
> params.filterByArea = True
> params.minArea = 200
> 
> # # 컬러, 볼록 비율, 원형 비율 필터 옵션 끄기
> params.filterByColor = False
> params.filterByConvexity = False
> params.filterByInertia = False
> params.filterByCircularity = False
> 
> # 필터 파라미터로 BLOB 검출기 생성
> detector = cv2.SimpleBlobDetector_create(params)
> 
> # 키 포인트 검출
> keypoints = detector.detect(gray)
> 
> # 키 포인트 그리기
> img_draw = cv2.drawKeypoints(img, keypoints, None, None,
>                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
> 
> # 결과 출력
> cv2.imshow("Blob with Params", img_draw)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 
> ![Untitled](8%202%20%E1%84%8B%E1%85%A7%E1%86%BC%E1%84%89%E1%85%A1%E1%86%BC%E1%84%8B%E1%85%B4%20%E1%84%90%E1%85%B3%E1%86%A8%E1%84%8C%E1%85%B5%E1%86%BC%E1%84%80%E1%85%AA%20%E1%84%8F%E1%85%B5%20%E1%84%91%E1%85%A9%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%90%E1%85%B3%20f338a659949443708d5e97803471b424/Untitled%205.png)
>