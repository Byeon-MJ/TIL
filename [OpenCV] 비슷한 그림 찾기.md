# 8.1 비슷한 그림 찾기

영상 속 객체를 인식하는 방법 중 하나로 비슷한 그림을 찾아내는 방법이 있다. 

비슷한 그림을 찾기 위한 여러가지 기술들 중에서 비교적 단순하고 간단한 것부터 알아본다.

### 8.1.1 평균 해시 매칭

쉽고 간단한 만큼 실효성은 떨어지긴하지만 비슷한 그림을 찾는 원리를 이해해난 데는 **평균 해시 매칭(average hash matching)** 이 좋다. 어떤 영상이든 동일한 크기의 하나의 숫자로 변환되는데, 이때 숫자를 얻기 위해서 평균값을 이용하는 방법이다. 평균을 얻기 위해서 영상을 특정한 크기로 축소하고, 픽셀 전체의 평균값을 구해서 각 픽셀의 값이 평균보다 작으면 0, 크면 1로 일괄 변환한다.

> **[예제 8-1] 권총을 평균 해시로 변환(8.1_avg_hash.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import matplotlib.pyplot as plt
> 
> # Read Image and convert gray scale
> img = cv2.imread('./img/pistol.jpg')
> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> 
> # Reduce to a size of 16 x 16
> gray = cv2.resize(gray, (16, 16))
> # Calculate the average value of a image
> avg = gray.mean()
> # Convert to 0 and 1 based on the mean value
> bin = 1 * (gray > avg)
> print(bin)
> 
> >>> [[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
> 		 [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
> 		 [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
> 		 [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
> 		 [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
> 		 [1 0 0 0 0 0 0 1 0 0 1 1 1 1 1 1]
> 		 [1 1 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
> 		 [1 1 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
> 		 [1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 1]
> 		 [1 1 0 0 0 0 1 1 1 1 1 1 1 1 1 1]
> 		 [1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1]
> 		 [1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1]
> 		 [1 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1]
> 		 [1 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1]
> 		 [1 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1]
> 		 [1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1]]
> 
> # Convert binary string to hexadecimal string
> dhash = []
> for row in bin.tolist():
>     s = ''.join([str(i) for i in row])
>     dhash.append(f'{int(s, 2):02x}')
> dhash = ''.join(dhash)
> print(dhash)
> 
> >>> ffff8000800080008000813fc1ffc1ffc07fc3ffc7ffc7ff87ff87ff87ffc7ff
> 
> plt.imshow(img[:,:,::-1])
> plt.title('pistol')
> plt.xticks([])
> plt.yticks([])
> plt.show()
> ```
> 
> ![Untitled](https://user-images.githubusercontent.com/69300448/229826308-5d579b0b-bf11-49dd-bfdb-6b2d27794719.png)
> 

위 방법으로 얻은 평균 해시를 다른 영상과 비교해서 얼마나 비슷한지 알아내야 한다.

유사도를 측정하기 위한 방법으로는 두 값의 거리를 측정해서 그 거리가 가까우면 비슷한 것으로 판단하는 방법을 이용한다.

- **유클리드 거리(Euclidian distance)**
    - 두 값의 차이로 거리 계산
    - 5와 비교할 값으로 8과 3이 있다면 5와 3의 유클리드 거리는 2, 5와 8의 유클리드 거리는 3이다. 따라서 5와 더 비슷한 수는 3이라고 판단
- **해밍 거리(Hamming distance)**
    - 두 수의 같은 자리의 값이 서로 다른 것이 몇 개인지를 나타내는 거리 계산 방법, 두 값의 길이가 같아야 계산이 가능하다.
    - 12345와 비교할 값으로 12354와 92345가 있을 때, 12345와 12354는 마지막 두 자리가 다르므로 해밍 거리는 2, 12345와 92345는 처음 자리 하나만 다르므로 해밍 거리는 1이다. 따라서 12345와 더 비슷한 수는 92345라고 판단

영상의 평균 해시값을 비교할 때 높은 자릿수가 다를수록 더 큰 거리로 인식하는 유클리드 거리보다는 각 자릿수의 차이의 개수로만 비교하는 해밍 거리로 측정하는 것이 더 적합하다. 

이처럼 숫자 2개를 비교할 때 그 수가 가지는 특징에 따라 어떤 방법으로 측정하는 것이 더 좋은지 판단해야한다. 앞의 두 가지 계산방식과 같은 기본 개념을 바탕으로 상황에 따라 숫자의 특성을 더 잘 살릴 수 있는 여러 가지 추가적인 연산 방법이 있다.

앞에서 구한 평균 해시를 이용해서 실제로 비슷한 이미지가 있는지 판단하기 위한 이미지 데이터셋이 필요하다. 미국의 캘리포니아 공대에서 제공하는 101가지 물체 이미지 셋을 이용하였다.

- 다운로드 페이지(라이센스 이유로 직접 다운로드 권장)

[Caltech 101](https://data.caltech.edu/records/mzrjq-6wc02)

> **[예제 8-2] 사물 영상 중에서 권총 영상 찾기(8.2_avg_hash_matching.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> import glob
> import matplotlib.pyplot as plt
> 
> # Read and display image
> img = cv2.imread('./img/pistol.jpg')
> # cv2.imshow('query', img)
> plt.imshow(img[:,:,::-1])
> plt.title('query')
> plt.xticks([])
> plt.yticks([])
> plt.show()
> 
> # Path to the images to be compared
> search_dir = './img/101_ObjectCategories'
> 
> # Convert the image to an average hash with a size of 16 x 16
> def img2hash(img):
>     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
>     gray = cv2.resize(gray, (16, 16))
>     avg = gray.mean()
>     bi = 1 * (gray > avg)
>     return bi
> 
> # Hamming distance measurement function
> def hamming_distance(a, b):
>     a = a.reshape(1, -1)
>     b = b.reshape(1, -1)
>     # Sum of values that are different from each other in the same position
>     distance = (a != b).sum()
>     return distance
> 
> # Get the hash of a pistol image
> query_hash = img2hash(img)
> 
> # Path of all image files in the image dataset directory
> img_path = glob.glob(search_dir+'/**/*.jpg')
> 
> img2 = cv2.imread('./img/pistol.jpg')
> cv2.imshow('query', img2)
> img_list = []
> img_list.append(img2)
> for path in img_path:
>     # Read and display one image from the dataset
>     img = cv2.imread(path)
>     cv2.imshow('searching...', img)
>     cv2.waitKey(5)
>     # Calculate the hash of one image from the dataset
>     a_hash = img2hash(img)
>     # Calculate the Hamming distance
>     dst = hamming_distance(query_hash, a_hash)
>     if dst/256 < 0.25:    # Print within the Hamming distance of 25%
>         print(path, dst/256)
>         img_list.append(img)
>         cv2.imshow(path, img)
> cv2.destroyWindow('searching...')
> cv2.waitKey()
> cv2.destroyAllWindows()
> 
> print(len(img_list))
> 
> >>> 20
> 
> for i, img in enumerate(img_list):
>     plt.subplot(4, 5, i+1)
>     if i == 0:
>         plt.title('original')
>     plt.imshow(img[:,:,::-1])
>     plt.xticks([])
>     plt.yticks([])
> plt.show()
> ```
> 
> ![Untitled 1](https://user-images.githubusercontent.com/69300448/229826351-59184fab-a265-4ea4-bb7e-aaa8bd2d2a8e.png)
> 

찾고자 하는 영상의 특징을 평균 해시라는 숫자 하나로 변환을 했다. 이와 같은 과정을 **특징 검출** 이라고 한다.

회전, 크기, 방향 등에 영향이 없으면서 정확도를 높이려면 특징을 잘 나타내는 여러 개의 지점을 찾아 그 특징을 잘 표현하고 서술할 수 있는 여러 개의 숫자들로 변환해야 하는데, 이것을 **키 포인트** 와 **특징 디스크립터(feature descriptor)** 라고 한다.

두 해시 값의 해밍 거리로 영상 간의 특징을 비교해서 유사도를 측정했는데, 이 과정은 **매칭(matching)** 이라고 한다. 특징점의 개수와 서술하는 방식에 따라 다양한 매칭 방법이 있다.

### 8.1.2 템플릿 매칭

어떤 물체가 있는 영상을 준비해두고 그 물체가 포함되어 있을 것이라고 예상할 수 있는 입력 영상과 비교해서 물체가 매칭되는 위치를 찾는 것을 **템플릿 매칭(template matching)** 이라고 한다. 미리 준비해둔 템플릿 영상은 입력 영상보다 크기가 항상 작아야 한다.

- **result = cv2.matchTemplate(img, templ, method[, result, mask])**
    - img : 입력 영상
    - templ : 템플릿 영상
    - method : 매칭 메서드
        - cv2.TM_SQDIFF : 제곱 차이 매칭, 완벽 매칭 : 0, 나쁜 매칭 : 큰 값
        
        $$
        R(x, y) = \sum_{x', y'}(T(x', y') - I(x + x', y + y'))^2
        $$
        
        - cv2.TM_SQDIFF_NORMED : 제곱 차이 매칭의 정규화
        - cv2.TM_CCORR : 상관관계 매칭, 완벽 매칭 : 큰 값, 나쁜 매칭 : 0
        
        $$
        R(x, y) = \sum_{x', y'}(T(x', y')
        \cdot I(x + x', y + y'))^2
        $$
        
        - cv2.TM_CCORR_NORMED : 상관관계 매칭의 정규화
        - cv2.TM_CCOEFF : 상관계수 매칭, 완벽 매칭 : 1, 나쁜 매칭 : -1
        
        $$
        R(x, y) = \sum_{x', y'}(T'(x', y') \cdot I'(x + x', y + y'))^2 \\
        T'(x', y') = T(x', y') - 1/(w \cdot h) \cdot \Sigma_{x'', y''}T(x'', y'') \\
        I'(x + x', y + y') = I(x + x', y + y') - 1/(w \cdot h) \cdot \Sigma_{x'', y''}I(x + x'', y + y'')
        $$
        
        - cv2.TM_CCOEFF_NORMED : 상관계수 매칭의 정규화
    - result : 매칭 결과, $(W - w + 1) \times (H - h + 1)$ 크기의 2차원 배열
        - $W, H$ : img의 열과 행
        - $w, h$ : tmpl의 열과 행
    - mask : TM_SQDIFF, TM_CCORR_NORMED인 경우 사용할 마스크
- **minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc( src[, mask] )**
    - src : 입력 1채널 배열
    - minVal, maxVal : 배열 전체에서 최소값, 최대값
    - minLoc, maxLoc : 최소값과 최대값의 좌표(x, y)

**cv2.matchTemplate()** 함수의 결과는 img의 크기에서 templ 크기를 뺀 것에 1만큼 큰 2차원 배열을 result로 얻는다. 이 배열의 최대, 최소값을 구하면 원하는 최선의 매칭값과 매칭점을 찾을 수 있다.

> **[예제 8-3] 템플릿 매칭으로 객체 위치 검출(8.3_template_matching.ipynb)**
> 
> 
> ```python
> import cv2
> import numpy as np
> 
> # Read input image and template image
> img = cv2.imread('./img/figures.jpg')
> template = cv2.imread('./img/taekwonv1.jpg')
> th, tw = template.shape[:2]
> 
> # Iterate three matching methods
> methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED']
> 
> cv2.imshow('template', template)
> for i, method_name in enumerate(methods):
>     img_draw = img.copy()
>     method = eval(method_name)
>     # Template matching
>     res = cv2.matchTemplate(img, template, method)
>     # Get the maximum value, minimum value, and their coordinates
>     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
>     print(method_name, min_val, max_val, min_loc, max_loc)
>     
>     # TM_SQDIFF : minimum value is the best matching, others : opposite
>     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
>         top_left = min_loc
>         match_val = min_val
>     else:
>         top_left = max_loc
>         match_val = max_val
>     # Get matching coordinates and display them in a rectangle
>     bottom_right = (top_left[0] + tw, top_left[1] + th)
>     cv2.rectangle(img_draw, top_left, bottom_right, (0, 0, 255), 2)
>     # Display matching point
>     cv2.putText(img_draw, str(match_val), top_left, cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, cv2.LINE_AA)
>     cv2.imshow(method_name, img_draw)
> cv2.waitKey()
> cv2.destroyAllWindows()
> ```
> 

템플릿 매칭은 크기, 방향, 회전 등의 변화에는 잘 검출되지 않고 속도가 느리다는 단점이 있다.
