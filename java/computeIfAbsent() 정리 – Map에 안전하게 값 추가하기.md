Java에서 `Map`에 값을 추가할 때, 특정 키가 없으면 새 객체를 생성해서 넣고, 그 객체를 바로 사용하는 패턴을 자주 사용하게 됩니다. 이 과정을 **깔끔하고 안전하게 처리**할 수 있도록 도와주는 메서드가 바로 `computeIfAbsent()`입니다.

---

## 기본 개념

```java
V computeIfAbsent(K key, Function<? super K, ? extends V> mappingFunction)
```

- `key`: 조회할 키
- `mappingFunction`: 키가 없을 때 값을 생성하는 함수
- 반환값: 존재하는 값 또는 새로 생성된 값

---

## 전통적인 방식 vs computeIfAbsent

### 🔸 기존 방식

```java
if (!map.containsKey(key)) {
    map.put(key, new ArrayList<>());
}
map.get(key).add("value");
```

### ✅ `computeIfAbsent()` 사용

```java
map.computeIfAbsent(key, k -> new ArrayList<>()).add("value");
```

불필요한 `if`와 `put`을 줄이고 코드가 훨씬 깔끔해졌습니다!

---

## 실전 예제: 날짜별 로그 집계

다음과 같이 날짜별 로그 집계를 저장하는 코드가 있다고 가정해 봅시다.

```java
Map<LocalDate, LogData> mergedData = new TreeMap<>();

LocalDate date = LocalDate.of(2025, 2, 11);
int total = 10;
int error = 3;

mergedData.computeIfAbsent(date, d -> new LogData()).add(total, error);
```

### 🔍 위 코드는 어떤 작업을 하나요?

1. `date`라는 키로 `mergedData` 맵을 조회
2. 키가 없으면 `new LogData()` 객체를 생성해서 맵에 추가
3. 그리고 `add()` 메서드를 호출해 로그 수치를 누적

### 🔸 `LogData` 클래스는 예를 들어 이렇게 생겼습니다:

```java
public class LogData {
    private int totalCount;
    private int errorCount;

    public void add(int total, int error) {
        this.totalCount += total;
        this.errorCount += error;
    }

    // getter/setter 생략
}
```

---

## ✅ if문으로 바꾸면?

```java
if (!mergedData.containsKey(date)) {
    mergedData.put(date, new LogData());
}
mergedData.get(date).add(total, error);
```

`computeIfAbsent()`는 이 모든 로직을 단 **한 줄**로 바꿔주는 강력한 유틸입니다.

---

## 비슷한 메서드와 비교

| 메서드 | 설명 |
| --- | --- |
| `computeIfAbsent()` | 키가 없을 때만 값 생성 |
| `compute()` | 항상 계산함 (있든 없든) |
| `merge()` | 값이 없으면 저장, 있으면 병합 로직 실행 |

---

## 마무리

`computeIfAbsent()`는 Java 8 이후부터 제공되는 함수형 스타일 API로,

**Map의 값을 생성/추가하는 로직을 매우 간결하고 안전하게** 만들어줍니다.

특히 `Map<K, List<V>>` 또는 `Map<K, CustomObject>` 같은 패턴에 자주 쓰입니다.

---

### ✨ 추천 활용 예시

- 날짜별/카테고리별 데이터 누적
- 키워드별 그룹핑
- 상태별 객체 관리

---
