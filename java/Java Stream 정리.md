Java 8부터 도입된 `Stream API`는 데이터를 처리하는 방식을 획기적으로 바뀌었다. 

기존의 반복문 기반 명령형 코드 대신, 선언형 방식으로 데이터를 **필터링, 매핑, 정렬, 수집**할 수 있게 해 준다.

Java Stream을 공부하면서 개념부터 핵심 기능, 실전 예제까지 정리

---

## Stream이란?

Stream은 컬렉션, 배열 등 **데이터 소스를 추상화**하여 연속적인 **데이터 처리 파이프라인**을 구성할 수 있도록 해주는 기능

Stream은 **데이터 자체를 저장하지 않으며**, **일회성이다.**

```java
List<String> names = Arrays.asList("Kim", "Lee", "Park");

names.stream()
     .filter(name -> name.startsWith("K"))
     .map(String::toUpperCase)
     .forEach(System.out::println);
```

---

## Stream의 특징

| 특징 | 설명 |
| --- | --- |
| 선언형 처리 | 루프 없이 데이터 흐름만 선언 |
| 체이닝 가능 | 연속된 중간 연산 구성 가능 |
| 내부 반복 | for-each 대신 내부적으로 반복 처리 |
| 일회성 | 한 번 사용하면 재사용 불가 |

---

## Stream의 구성

### 1. 스트림 생성 (Source)

```java
// List 기반
Stream<String> stream1 = list.stream();

// 배열 기반
Stream<Integer> stream2 = Stream.of(1, 2, 3);

// 무한 스트림
Stream<Double> infinite = Stream.generate(Math::random);
```

### 2. 중간 연산 (Intermediate Operation)

중간 연산은 **스트림을 가공**하지만 실행은 되지 않는다. 최종 연산이 호출되어야 실행

| 메서드 | 설명 |
| --- | --- |
| `filter()` | 조건 필터링 |
| `map()` | 요소 변환 |
| `sorted()` | 정렬 |
| `distinct()` | 중복 제거 |
| `limit(n)` | n개 제한 |
| `skip(n)` | 앞에서 n개 건너뜀 |

```java
list.stream()
    .filter(s -> s.length() > 1)
    .map(String::toUpperCase)
    .sorted();
```

### 3. 최종 연산 (Terminal Operation)

최종 연산은 **실제 결과를 생성**하며, 이때 스트림이 소비됨.

| 메서드 | 설명 |
| --- | --- |
| `forEach()` | 반복 처리 |
| `collect()` | 결과 수집 |
| `count()` | 요소 수 반환 |
| `anyMatch()` 등 | 조건 일치 여부 |
| `reduce()` | 누적 연산 |

```java
List<String> result = list.stream()
                          .map(String::toLowerCase)
                          .collect(Collectors.toList());
```

---

## 실전 예제

### 1. 특정 문자로 시작하는 요소 수 세기

```java
long count = list.stream()
                 .filter(s -> s.startsWith("K"))
                 .count();
```

### 2. 짝수만 제곱하여 리스트로 만들기

```java
List<Integer> result = numbers.stream()
                              .filter(n -> n % 2 == 0)
                              .map(n -> n * n)
                              .collect(Collectors.toList());
```

### 3. reduce로 총합 구하기

```java
int sum = numbers.stream()
                 .reduce(0, Integer::sum);
```

---

## 병렬 스트림 (Parallel Stream)

멀티코어 환경에서 성능 향상을 위해 `parallelStream()`을 사용할 수 있다.

```java
list.parallelStream()
    .filter(...)
    .map(...)
    .forEach(...);
```

주의: 순서가 중요한 연산에는 병렬 스트림 사용을 피하자!

---

## Stream vs for-each 비교

| 항목 | for-each | Stream |
| --- | --- | --- |
| 코드 스타일 | 명령형 | 선언형 |
| 코드 길이 | 길어질 수 있음 | 간결함 |
| 병렬 처리 | 직접 구현 필요 | `parallelStream()` 사용 가능 |
| 상태 관리 | 명시적 | 암묵적 (불변 기반) |

---

## 디버깅 팁: peek() 사용

Stream의 중간 상태를 확인하고 싶을 때는 `peek()`을 활용 가능

```java
list.stream()
    .peek(System.out::println)
    .filter(...)
    .collect(...);
```

---

## 마무리

- Java Stream을 잘 활용하지 못해서 아직은 낯설다.
- 익숙해지면 반복문보다 훨씬 **직관적이고 유지보수가 쉬운 코드**를 작성할 수 있을 것 같다.

---

**Tags:** `Java`, `Stream`, `Functional Programming`, `Java8`, `개발팁`, `함수형 프로그래밍`
