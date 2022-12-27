# [Python] 자료구조 : Hash

Hash(해시) 자료구조에 대한 공부!

코딩 테스트 등에서 출제 빈도가 높다고 해서 따로 한번 정리해 보았다.

# Hash?

- 대표적인 자료 구조 중 하나로써, Key & Value로 구성되어있어 데이터 검색과 삽입, 추출 등의 작업에서 빠른 속도로 작업을 완수할 수 있다.
- 해시를 쓰지않고 리스트와 같은 자료형을 사용할 경우 전체 자료구조를 검색하기때문에 효율성이 떨어진다.
- 파이썬에서는 `Dictionary` 자료구조가 Hash 형태로 구현되어 있다.

# Hash 사용하기

해시가 빠르다는건 알겠다. 그렇다면 언제 사용하면 좋을까?

## 1. 리스트를 사용할수 없는 경우

리스트는 인덱스를 접근할때 숫자 인덱스를 이용하여 접근한다. 하지만 인덱스 값이 숫자가 아닌 문자열이라면? 이럴 때에는 리스트보다는 딕셔너리를 사용하는 것이 좋다.

## 2. 빠른 검색 / 추출 등이 필요할 때

위에서도 말했지만 해시 자료구조는 속도적인 면에서 리스트보다 훨씬 빠르기 때문에 빠른 속도가 필요할때 사용하면 좋다.

### Hash 와 List 의 시간복잡도 비교

| Operation | Dictionary | List |
| --- | --- | --- |
| Get Item | O(1) | O(1) |
| Insert Item | O(1) | O(1) ~ O(N) |
| Update Item | O(1) | O(1) |
| Delete Item | O(1) | O(1) ~ O(N) |
| Search Item | O(1) | O(N) |
| Index | O(1) | O(1) |
| Store | O(1) | O(1) |
| Length | O(1) | O(1) |
| Pop | O(1) | O(N) |
| Remove | - | O(N) |
| Sort | - | O(N Log N) |
| Iteration | O(N) | O(N) |
| Check = =, ! = | - | O(N) |

## 3.  집계가 필요 할 때

원소의 갯수를 세거나 차이를 비교할 때, 해시와 collections 모듈의 Counter 클래스 등을 사용하면 훨씬 빠르게 문제를 해결할 수 있다.

# 사용 예

딕셔너리의 자세한 사용법은 생략하고 프로그래머스 예제와 함께 List와 비교해보도록 하자

- 문제

[코딩테스트 연습 - 완주하지 못한 선수](https://school.programmers.co.kr/learn/courses/30/lessons/42576)

- 입출력 예

![Untitled](%5BPython%5D%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%20Hash%2038af85ecf13b48b08ffd7ce003cba7b0/Untitled.png)

입출력 예처럼 `participant` 와 `completion`이라는 두 리스트를 비교해서 

 `participant` 에는 있지만 `completion`에는 없는 원소값을 리턴하는 문제이다.

### List를 이용한 문제 풀이

```python
def solution(participant, completion):
    for i in completion:
        participant.remove(i)
    return participant[0]
```

처음에는 위와 같이 List를 이용하여 문제풀이를 진행했다. 

전제 리스트 길이만큼 반복하여 탐색하기 때문에 시간이 오래 걸리고, 결과적으로 효율성 테스트에서 시간초과가 걸려서 문제를 통과하지 못했다.

![Untitled](%5BPython%5D%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%20Hash%2038af85ecf13b48b08ffd7ce003cba7b0/Untitled%201.png)

### Dictionary를 이용한 문제 풀이

```python
def solution(participant, completion):
    dic = {}
    for p in participant:
        if p not in dic: # key
            dic[p] = 0
        dic[p] += 1
    for c in completion:
        dic[c] -= 1
        if dic[c] == 0:
            del dic[c]
    return list(dic.keys())[0]
```

딕셔너리를 이용하여 문제를 풀이해주면 효율성 테스트도 통과하여 정답을 맞출수 있었다.

![Untitled](%5BPython%5D%20%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%20Hash%2038af85ecf13b48b08ffd7ce003cba7b0/Untitled%202.png)

앞으로 원소를 검색하거나 더 빠른 속도로 문제를 해결해야 할 때에는 해시를 기억하고 활용해보도록 해야겠다!

### Hash, Dictionary에 대한 추가 자료

[딕셔너리(Dictionary)](https://yunaaaas.tistory.com/2)

[리스트(List) / 딕셔너리(Dictionary) 정렬](https://yunaaaas.tistory.com/5)

## Reference

[TimeComplexity - Python Wiki](https://wiki.python.org/moin/TimeComplexity)

[[Python 자료구조] Hash(해시)](https://yunaaaas.tistory.com/46)

[[Python 파이썬] 해시 (Hash) 해싱 (Hashing) 알고리즘 예제로 알아보기](https://codingpractices.tistory.com/entry/Python-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%ED%95%B4%EC%8B%9C-Hash-%ED%95%B4%EC%8B%B1-Hashing-%EB%AC%B8%EC%A0%9C%EB%A1%9C-%EC%95%8C%EC%95%84%EB%B3%B4%EA%B8%B0)

[[Python] List, Dict 시간 복잡도 (Big O)](https://gomguard.tistory.com/181)