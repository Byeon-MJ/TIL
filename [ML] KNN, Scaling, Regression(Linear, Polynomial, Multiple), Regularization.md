# [ML] KNN, Scaling, Regression(Linear, Polynomial, Multiple), Regularization

[Notion 정리 링크](https://stump-python-602.notion.site/ML-KNN-Scaling-Regression-Linear-Polynomial-Multiple-Regularization-27add895f71e4e0b9788a4409b4912e2)
# K-최근접 이웃(KNN)

- 지도 학습 알고리즘 중 하나, 굉장히 직관적이고 간단하다.
- 어떤 데이터가 주어지면 그 주변(이웃)의 데이터를 살펴본 뒤 더 많은 데이터가 포함되어 있는 범주로 분류하는 방식
- KNN은 분류와 회귀 문제 둘다 사용할 수 있다.

## KNN Classification

![출처: towardsdatascience](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/223643b5-0c01-4c7b-be59-9427df35fb1c/Untitled.png)

출처: towardsdatascience

- 새로운 데이터(빨간 점) 이 주어졌을 때 이를 A로 분류할지, B로 분류할지 판단하는 문제
- k=3(안쪽 원)일 때, 노란색 점(Class A) 1개와 보라색 점(Class B) 2개가 있다. 따라서 k=3일 때 빨간 점은 B로 분류 된다.
- k=6(바깥쪽 원)일 때, 노란색 점 4개와 보라색 점 2개가 있으므로 노란색 점으로 분류된다.
- K의 값에 따라 결과가 달라지므로 너무 작아서도 안되고 너무 커서도 안된다.
- 보통 K 값은 홀수를 사용한다. 짝수일 경우 동점이 발생할 수가 있어 결과를 도출할 수 없기 때문이다.

## 거리 계산

- KNN 에서는 데이터와 데이터 사이의 거리에 따라 결과가 달라질 수 있다.
- 데이터의 스케일에 따라 큰 영향을 받기 때문에 데이터 전처리가 필요함

### 1. Euclidean Distance

일반적인 점과 점사이의 거리를 구하는 방법이다.

![출처: Wikipedia](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3b197e10-c048-4583-baa5-6916f1b5ca3f/Untitled.png)

출처: Wikipedia

3차원에서 유클리드 거리 구하는 예시

![출처: ratsgo's blog](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/82ec1e39-cd1c-4fd1-bf17-6a8cccc3e954/Untitled.png)

출처: ratsgo's blog

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/226ee6a5-2deb-4768-814f-55f870b351e7/Untitled.png)

### 2. Manhattan Distance

점과 점 사이의 직선 거리가 아니라 X축과 Yㅊ푹을 따라 간 거리를 의미한다.

아래 예시의 Route 1, Route 2, Route 3은 서로 다른 길이지만 맨해튼 거리로는 같다.

## KNN Regression

- KNN 회귀 모델은 분류와 동일하게 가까운 k개의 이웃을 찾는다.
- 그 다음 이웃 샘플 타깃값의 평균을 구하여 예측값으로 사용한다.
- Sckikit Learn은 회귀 모델의 점수로 결정계수 값을 반환한다.

# Data Scaling

- 데이터 간의 거리에 영향을 받는 알고리즘을 사용할 때, 데이터의 스케일을 일정한 기준으로 맞춰 주는 방법
- 크게 Standardizaion(표준화) 와 Normalization(정규화) 가 있음
    
    

## Standardization

- 특성들의 평균을 0, 분산을 1로 스케일링

### z-Score Standardization(Numpy)

- 표준점수를 활용해 Numpy로 표준화하기

```python
mean = mp.mean(train_input, axis=0)
std = mp.std(train_imput, axis=0)

train_scaled = (train_input - mean) / std
```

## Normalization

- 특성들을 특정 범위(주로 0~1)로 스케일링 하는 것

## Sickit Learn Scaler

- scaler 는 fit 과 transform 메서드를 가진다
- fit 메서드는 훈련 데이터에 적용해서 훈련 데이터의 분포를 먼저 학습
- tranform 메서드를 이용해 훈련 데이터와 테스트 데이터에 적용
- fit 과 transform이 결합된 fit_transform() 메서드가 있다. 훈련데이터에만 적용할 것
- 특성에 따라 각기 다른 스케일링을 적용하는 것이 유리할 수 있음

### StandardScaler()

- 특성들의 평균을 0, 분산을 1로 스케일링
- 최솟값과 최댓값의 크기를 제한하지 않기때문에 특정 알고리즘에서는 문제가 있을 수 있음
- **이상치에 매우 민감**
- 회귀보다는 분류에 유용하다

### MinMaxScaler()

- 가장 작은 값이 0, 가장 큰 값이 1로 변환되므로, 모든 특성들이 [0, 1]의 범위를 갖게 스케일링
- **이상치에 매우 민감**
- 분류보다 회귀에 유용하다

### ****MaxAbsScaler()****

- 각 특성의 절대값이 0과 1사이가 되도록 스케일링
- 모든 값이 -1 ~ 1의 값으로 표현된다
- 데이터가 모두 양수일 경우에는 MinMaxScaler와 같다.
- **이상치에 매우 민감**

### ****RobustScaler()****

- 평균과 분산 대신에 중앙값과 사분위값을 사용
- **이상치의 영향을 최소화**

### Normalizer()

- Normalizer 의 경우 각 샘플(행)마다 적용되는 방식
- 한 행의 모든 특성들 사이의 유클리드 거리가 1이 되도록 스케일링
- 일반적인 데이터 전처리의 상황에서 사용되는 것이 아니라 모델(특히나 딥러닝) 내 학습 벡터에 적용

# Linear Regression

- 독립변수도 하나, 종속변수도 하나인 단순한 형태의 선형 회귀
- 최적의 직선을 찾는 알고리즘

# Polynomial Regression

- 직선이 데이터의 특성을 알맞게 표현하지 못하고 최적의 곡선을 찾아야 할때
- 항을 추가하여 다항식의 형태를 갖게 되는 선형 회귀

# Multiple Regression

- 여러 개의 특성을 사용한 선형 회귀
- 특성이 2개면 선형회귀는 평면을 학습한다.

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


### L2

- 상대적으로 큰 회귀 계수 값의 예측 영향도를 감소시키기 위해서 회귀 계수값을 더 작게 만드는 규제 모델


## Ridge & Lasso…Elastic-net

- Lasso
    
    L1 규제를 추가한 회귀 모델
    
- Ridge
    
    L2 규제를 추가한 회귀 모델
    
- Elastic-net
    
    릿지와 라쏘를 결합한 모델
    
    주로 특성이 많은 데이터 세트에 적용
    

## Hyper Parameter

- 머신러닝 알고리즘이 학습하지 않는 파라미터, 사람이 직접 지정해줘야 한다.
- Ridge와 Lasso의 alpha 파라미터가 대표적

## Reference
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing<br>
https://justweon-dev.tistory.com/19<br>
https://wooono.tistory.com/96
