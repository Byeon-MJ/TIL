# [ML] Logistic Regression

# 로지스틱 회귀란?

- **로지스틱 회귀**는 선형 회귀 방식을 분류에 적용한 알고리즘이다.
- 머신러닝에서 **2진 분류(Binary Classification)** 모델로 사용된다. 다중분류도 가능은 함.
- 데이터가 어떤 범주에 속할 확률을 0에서 1 사이의 값으로 예측하고 그 확률에 따라 가능성이 더 높은 범주에 속하는 것으로 분류한다.
- **선형 회귀와 차이점** : 학습을 통해 선형 함수의 회귀 최적선을 찾는 것이 아니라 시그모이드(Sigmoid) 함수 최적선을 찾고 이 시그모이드 함수의 반환 값을 확률로 간주해 분류를 걸정한다

![Untitled](https://user-images.githubusercontent.com/69300448/211442401-ec60796a-3228-4ba8-af50-082d16340710.png)

# 로지스틱 회귀의 확률 추정

- 시그모이드 함수의 $x$값이 아무리 커지거나 작아져도 $y$는  0과 1 사이의 값을 반환한다.
- 다중분류에서는 소프트맥스 함수를 사용하여 확률 반환

**시그모이드 함수 정의**

$$
y = \frac{1}{1+ e^{-t}}
$$

![Untitled 1](https://user-images.githubusercontent.com/69300448/211442457-7cf3472f-e568-4789-ae3c-5ee393109471.png)

- 위 계산 식에서 t를 `logit` 혹은 log-odds 라고 부른다. 양성, 음성 클래스의 추정확률 사이의 로그 비율
- 각 feature들의 계수 log-odds를 구한 후 Sigmoid 함수를 적용하여, 실제로 데이터가 해당 클래스에 속할 확률을 0과 1사이의 값으로 나타낸다.
- 해당 클래스인지 분류되는 추정 확률은 Threshold에 의해 결정된다. 기본 값은 0.5(50%)이지만 데이터의 특성이나 상황에 따라 조정할 수 있다.
- Scikit-learn을 통해 모델을 생성하고 각 속성(feature)들의 계수를 구할 수 있다. 각 계수(coefficients)들은 데이터를 분류함에 있어 해당 속성이 얼마나 중요한지 해석하는 데에 사용할 수 있다.

# 훈련과 비용 함수

- 로지스틱 회귀는 y = 1인 양성 샘플에 대해서는 높은 확률을 추정하고 음성 샘플에 대해서는 낮은 확률을 추정하는 모델의 파라미터 벡터를 찾는 것
- log-odds 가 0에 가까워지면 -log(t)가 커지고,  log-odds가 1에 가까우면 -log(t)는 0에 가까워진다.
- 전체 훈련세트에 대한 비용함수는 훈련 샘플의 비용을 평균한 것 → log loss
    
$$
J(\theta) = -{1 \over m} \sum_{i=1}^{m} \lbrack y^{(i)}log(\hat{p}^{(i)})+(1-y^{(i)})log(1-\hat{p}^{(i)}) \rbrack
$$
    
    - `m`:데이터 총 개수
    - `y_i`: 데이터 샘플 `i`의 분류
    - `i`: 데이터 샘플 log-odd
    - `p_i`: 데이터 샘플 `i`의 log-odd의 sigmoid (즉, 데이터 샘플 `i`가 분류에 속할 확률)
- log loss를 활용하여 편미분 하면 경사하강법 알고리즘을 사용할 수 있다.

# Reference
[Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow](https://m.hanbit.co.kr/store/books/book_view.html?p_code=B7033438574)
[hleecaster.com](https://hleecaster.com/ml-logistic-regression-concept/)

[[딥러닝] 선형회귀와 로지스틱회귀](https://ebbnflow.tistory.com/129)

[[혼공머] Logistic Regression(로지스틱 회귀)](https://velog.io/@cha-suyeon/%ED%98%BC%EA%B3%B5%EB%A8%B8-Logistic-Regression%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1-%ED%9A%8C%EA%B7%80)
