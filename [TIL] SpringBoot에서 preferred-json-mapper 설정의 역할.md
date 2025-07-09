## Intro…

갑자기 멀쩡하던 gson 라이브러리 쪽에서 NullPointerException이 발생했다.

문제를 해결하기 위해 라이브러리 의존성이 잘 추가되었나 확인도 하고, 빌드 후 라이브러리가 제대로 있는지도 확인하고 여러가지 확인을 했는데 에러의 원인을 찾을 수 없었다.

한참동안 고민하다가 발견한 것.  yml에서 spring.mvc.converters.preferred-json-mapper 설정.

해당 설정이 누락된 것을 발견하고 추가한 뒤 다시 테스트. 다행히 정상 동작!!

그동안 yml(또는 properties) 설정에서 기존 설정값의 역할을 이해하지 못하고 그냥 사용하기만 했었는데, 이번 경험을 계기로 SpringBoot config 설정들에 대해서 최소한 어떤 역할을 하는 것인지는 알아야겠다고 생각했다. 

그래서 정리하는 첫번째. preferred-json-mapper

---

## 📌 개요

- Spring Boot는 기본적으로 JSON 직렬화/역직렬화에 Jackson 라이브러리를 사용
- 프로젝트에 따라 Gson을 사용할 수 있음
- Spring Boot에서는 `application.yml` 설정만으로 기본 JSON Mapper를 Gson으로 간편하게 변경 가능

---

## 🔧 설정 방법

```yaml
spring:
  mvc:
    converters:
      preferred-json-mapper: gson
```

위 설정을 `application.yml`에 추가하면 Spring MVC는 Jackson 대신 **Gson**을 기본 JSON 직렬화 도구로 사용하게 됨

---

## ✅ 적용 효과

- `@RestController`의 리턴 객체를 Gson을 통해 JSON으로 직렬화
- `@RequestBody`로 전달받은 JSON 요청 데이터를 Gson을 통해 객체로 역직렬화

예:

```java
@RestController
public class HelloController {
    @GetMapping("/hello")
    public MyDto hello() {
        return new MyDto("Hello", 123);
    }
}
```

→ `/hello` 응답은 Gson 기반으로 직렬화된 JSON 형태로 반환됨

---

## 📦 의존성 추가 (Gradle)

Gson은 기본적으로 Spring Boot에 포함되지 않기 때문에 의존성을 직접 추가해야 함

```groovy
implementation 'com.google.code.gson:gson'
```

> 참고: Jackson 의존성이 함께 있어도 preferred-json-mapper: gson 설정이 있으면 Gson이 우선 적용
> 

---

## ⚠️ 주의할 점

| 항목 | Jackson | Gson |
| --- | --- | --- |
| Spring 기본 지원 | O | X |
| 커스터마이징 | 매우 유연 | 상대적으로 제한적 |
| 속성 필터링 (@JsonView 등) | 지원 | 미지원 |
| Null 처리 | 커스터마이징 가능 | 기본 동작 단순 |
| 성능 | 빠름 | 비교적 느림 (큰 차이는 아님) |

→ 복잡한 JSON 제어가 필요한 경우에는 Jackson이 더 적합

→ 간단하고 가벼운 직렬화가 필요할 경우에는 Gson도 충분히 유용

---

## ✅ 결론

- `preferred-json-mapper: gson` 설정을 통해 Spring Boot에서도 간편하게 Gson을 사용할 수 있음
- 의존성만 추가하면 별도 Bean 설정 없이도 Gson을 기본 직렬화기로 지정 가능
- 단, Jackson에 비해 기능이 제한적이므로 프로젝트 요구사항에 맞춰 선택해야 함

---

## 🔗 참고

- [Spring Boot 공식 문서](https://docs.spring.io/spring-boot/docs/current/reference/html/application-properties.html#application-properties.web.spring.mvc.converters.preferred-json-mapper)
- [Gson GitHub](https://github.com/google/gson)
