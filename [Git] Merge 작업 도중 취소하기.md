Git을 이용해 Branch를 나누고 코드를 관리하다보면, Merge는 꼭 사용하게 된다.

오늘도 Merge를 진행하던 도중, 실수로 다른 Branch와 Merge를 진행하였다. 심지어 Conflict까지 발생하였기 때문에 해당 Merge 작업 자체를 취소할 필요가 있었다.

Git Merge를 진행하는 중 Merge conflict가 발생할 경우, 해당 Merge 과정을 중지하고 Merge 작업 이전의 상태(pre-merge) 로 되돌리는 방법이다. 이 때, `--abort` 옵션을 사용한다. 
Merge가 완료된 이후에는 수행할 수 없다.

```jsx
git merge --abort
```
