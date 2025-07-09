# Linear Regression(Linear, Polynomial, Multiple)

# 선형 회귀란?

- 하나의 종속변수와 하나 이상의 독립변수 사이에 최적의 직선을 찾는 알고리즘
- 예측값과 실제값의 오차를 최소화하는 선을 찾아서 모델을 최적화
- 수식을 도출하기에 매우 쉽게 때문에 그 해석도 직관적

# 단순 선형 회귀(Linear Regression)

- 독립변수도 하나, 종속변수도 하나인 단순한 형태의 선형 회귀
    
    $$
    y = w_1*x + b
    $$
    
- 단순 선형 회귀에서는 기울기 $w_1$과 절편 $w_0$을 회귀 계수로 지칭하고 구한다.
- 아래 그래프는 생선의 길이와 무게 데이터로 선형 회귀선을 구해 본 것이다.

![Untitled](https://user-images.githubusercontent.com/69300448/209596011-41db4181-745d-4cd4-8d2a-6a7548bd118f.png)

# 다항 선형 회귀(Polynomial Regression)

- 직선이 데이터의 특성을 알맞게 표현하지 못하고 최적의 곡선을 찾아야 할 때
    
    $$
    y = w_1*x^2 + w_2*x + b
    $$
    
- 항을 추가하여 다항식의 형태를 갖게 되는 선형 회귀

![Untitled 1](https://user-images.githubusercontent.com/69300448/209596027-c4d8f8c3-7e9c-49a6-b5ff-b08c2c44a0b5.png)

# 다중 선형 회귀(Multiple Regression)

- 여러 개의 특성(독립 변수)와 하나의 종속변수를 사용한 선형 회귀 모델
    
    $$
    y = w_1*x_1 + w_2*x_2 + ... + w_n*x_n + b
    $$
    
- 특성이 2개면 선형회귀는 평면을 학습한다.
- 다중 선형 회귀에서는 다음과 같은 가정이 필요하다.
    1. 각각의 독립 변수는 종속 변수와의 선형 관계가 존재
    2. 독립 변수 사이에서는 높은 수준의 상관관계가 존재하지 않아야 함 - 다중 공선성 문제
    3. 추정된 종속 변수의 값과 실제 관찰된 종속 변수의 값과의 차이, 즉 잔차(residual)가 정규 분포를 이루어야 함
    

## Overfitting, Underfitting

- 모델의 훈련이 끝나면 훈련 세트와 테스트 세트에 대해서 평가 점수를 구할 수 있다.

### Overfitting

- 훈련 세트의 점수에 비해 테스트 세트의 점수가 과도하게 낮으면 **Overfitting**
- 훈련 데이터의 잡음의 양에 비해 모델이 너무 복잡할 때 발생
- 해결 방법
    - 파라미터 수가 적은 모델을 선택, 훈련 데이터의 특성의 수 줄이기
    - 모델이 훈련세트를 너무 과도하게 학습하지 못하도록 제약을 가하여 단순화시킴
    - 더 많은 훈련 데이터 사용해보기
    - 훈련 데이터의 잡음 줄이기(ex : 오류 데이터의 수정 or 이상치 제거 등)

### Underfitting

- 훈련 세트의 점수에 비해 테스트 세트의 점수가 너무 높거나 두 점수가 모두 너무 낮으면 **Underfitting**
- 모델이 너무 단순해서 데이터의 내재적인 구조를 모델이 학습하지 못하는 경우
- 해결 방법
    - 모델 파라미터가 더 많은 강력한 모델 선택
    - 학습 알고리즘에 더 좋은 특성 제공
    - 모델의 제약을 줄임

## Fiture Engineering

- 기존의 특성을 사용해 새로운 특성을 뽑아내는 작업

## Regularization

### L1

- 예측 영향력이 작은 특성의 회귀 계수를 0으로 만들어 회귀 예측시 특성이 선택되지 않게 하는 규제 모델

![Untitled 2](https://user-images.githubusercontent.com/69300448/209596059-97bc1e77-7127-4ce3-a09b-9762ce5ce130.png)

### L2

- 상대적으로 큰 회귀 계수 값의 예측 영향도를 감소시키기 위해서 회귀 계수값을 더 작게 만드는 규제 모델

![Untitled 3](https://user-images.githubusercontent.com/69300448/209596067-a307157a-2216-4911-b75d-f691b0948182.png)

## Ridge & Lasso…Elastic-net

- **Lasso**
    
    L1 규제를 추가한 회귀 모델
    
    상대적으로 큰 회귀 계수 값의 예측 영향도를 감소시키기 위해서 회귀 계수값을 더 작게 만드는 규제 모델
    
- **Ridge**
    
    L2 규제를 추가한 회귀 모델
    
    예측 영향력이 작은 피처의 회귀 계수를 0으로 만들어 회귀 예측 시 피처가 선택되지 않게 하는 것
    
- **Elastic-net**
    
    릿지와 라쏘를 결합한 모델
    
    주로 특성이 많은 데이터 세트에 적용, L1 규제로 피처의 개수를 줄임과 동시에 L2 규제로 계수 값의 크기를 조정
    
    
## Hyper Parameter

- 머신러닝 알고리즘이 학습하지 않는 파라미터, 사람이 직접 지정해줘야 한다.
- Ridge와 Lasso의 alpha 파라미터가 대표적
    

## Reference

[https://wikibook.co.kr/pymlrev2/](https://wikibook.co.kr/pymlrev2/)

[https://hongong.hanbit.co.kr/혼자-공부하는-머신러닝-딥러닝/](https://hongong.hanbit.co.kr/%ED%98%BC%EC%9E%90-%EA%B3%B5%EB%B6%80%ED%95%98%EB%8A%94-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D/)

[https://hleecaster.com/ml-linear-regression-concept/](https://hleecaster.com/ml-linear-regression-concept/)

[https://realblack0.github.io/2020/03/27/linear-regression.html](https://realblack0.github.io/2020/03/27/linear-regression.html)

[https://rebro.kr/187](https://rebro.kr/187)

[https://justweon-dev.tistory.com/19](https://justweon-dev.tistory.com/19)
