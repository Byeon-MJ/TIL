# [OpenCV] Error : OpenCV(4.6.0) :-1: error: (-5:Bad argument) in function 'add'

OpenCV 의 Tracker 를 사용하여 Face Tracking 프로젝트를 연습하던 중에 두가지 오류가 발생했다.

결론적으로는 둘다 같은 오류였고, 버전 차이에서 발생한 오류였다.

참고한 코드의 OpenCV version은 `4.1.2` 버전이었고 내가 실습한 Colab 환경의 OpenCV 버전은 `4.6.0` 버전이었다.

# 1. MultiTracker_create

![Untitled](%5BOpenCV%5D%20Error%20OpenCV(4%206%200)%20-1%20error%20(-5%20Bad%20argu%202c5d84204dd84a9597b5fbee32e53a8e/Untitled.png)

연습을 참고하며 공부한 코드에서는 `cv2.MultiTracker_create()` 를 이용해서 MultiTracker 를 불러왔는데,

> module ‘cv2’ has no attribute ‘MultiTracker_create’
> 

cv2에는 MultiTracker_create 속성이 없다는 에러가 발생했다.  OpenCV 문서와 StackOverFlow를 검색해보니 `4.5.1` 버전 이후에서는 `cv2.legacy.MultiTracker_create` 를 사용해주어야 했다.

# 2. trackers.add( )

```python
# trackers 객체가 설정된 tracker mode로 boxing된 영역 tracking
trackers.add(tracker, frame, (x, y, w, h))
```

MultiTracker_create 문제를 해결하고 코드를 진행하던 중, MultiTracker에 new Tracker를 add 하는 부분에서 또 에러가 발생했다. 에러 원인은 아래 내용.

error: OpenCV(4.6.0) :-1: error: (-5:Bad argument) in function 'add'

> Overload resolution failed:
> 
> - Expected Ptrcv::legacy::Tracker for argument 'newTracker'
> - Expected Ptrcv::legacy::Tracker for argument 'newTracker'

![Untitled](%5BOpenCV%5D%20Error%20OpenCV(4%206%200)%20-1%20error%20(-5%20Bad%20argu%202c5d84204dd84a9597b5fbee32e53a8e/Untitled%201.png)

add의 첫 파라미터로 new Tracker를 넣어줘야하는데 최초 트래커는 아래 코드와 같이 만들었었다. 

```python
# csrt
# tracker = cv2.TrackerCSRT_create()
# kcf
# tracker = cv2.TrackerKCF_create()
# boosting
# tracker = cv2.TrackerBoosting_create()
# mil
# tracker = cv2.TrackerMIL_create()
# tld
# tracker = cv2.TrackerTLD_create()
# medianflow
# tracker = cv2.TrackerMedianFlow_create()
# mosse
# tracker = cv2.TrackerMOSSE_create()
```

이것 또한 새로운 버전에 맞게 legacy를 불러오고 새로운 Tracker를 만들어 주니 잘 해결되었다.

```python
## OpenCV version 4.5.1
# csrt
# tracker = cv2.legacy.TrackerCSRT_create()
# kcf
tracker = cv2.legacy.TrackerKCF_create()
# boosting
# tracker = cv2.legacy.TrackerBoosting_create()
# mil
# tracker = cv2.legacy.TrackerMIL_create()
# tld
# tracker = cv2.legacy.TrackerTLD_create()
# medianflow
# tracker = cv2.legacy.TrackerMedianFlow_create()
# mosse
# tracker = cv2.legacy.TrackerMOSSE_create()
```

같은 내용의 에러를 찾기가 쉽지 않아서 많은 검색을 했고…Bounding Box가 잘못된 것인줄 알고 이것저것 많이 바꿔보았지만 해결이 되지않았다. 오전 내내 검색해보고 OpenCV 문서를 보다가 내가 보던 문서가 버전이 맞지 않는 것을 확인했고, 내가 사용하는 버전의 문서를 확인하고 오류를 수정할 수 있었다. Reference Document를 확인할 때 버전도 잘 체크해야겠다.

## Reference

[OpenCV: Legacy Tracking API](https://docs.opencv.org/4.6.0/dc/d6b/group__tracking__legacy.html)

[Attribute Error: MultiTracker_create() Not Found in cv2 on Raspberry Pi](https://stackoverflow.com/questions/54013403/attribute-error-multitracker-create-not-found-in-cv2-on-raspberry-pi)