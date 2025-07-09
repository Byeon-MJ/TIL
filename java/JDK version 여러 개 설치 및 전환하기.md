# [Java] JDK version 여러 개 설치 및 전환

회사에서 일을 하면서 그때그때 JDK 버전을 다르게 해서 사용해야 할 필요가 있었다.

특히 SpringBoot 3.0 이상의 버전을 사용하려면 Java17 이상을 사용해야 하기 때문에 Java 버전의 전환은 필수처럼 느껴졌다. 그래서 구글링을 통해 JDK 변환을 쉽게 하면서 사용하는 방법을 찾아보았다.

현재 나는 JDK 1.7, 1.8, 17 버전 3개를 설치해서 전환하면서 사용 중이다.

Java 8과 Java17을 설치하고 script 작성을 통해 두 개를 전환해가면서 사용하는 방법을 알아보자.

# 1. 필요한 Java(JDK) version 설치하기

- 우선 원하는 Java version을 설치해야 한다.
- 앞서 언급한 것처럼 Java 8과 17을 설치한다. 그리고 두 JDK의 경로는 한 폴더로 한다.

![Untitled](https://github.com/Byeon-MJ/TIL/assets/69300448/5c6ff9ba-bc2a-4b03-8f6c-25141d69161a)

# 2. scripts 폴더 생성 및 Java 버전 수에 맞는 .bat 파일 생성

- Java version 변경을 위한 script를 생성할 scripts 폴더를 만든다.
- 변환할 JDK의 수에 맞게 .bat 로 이루어진 파일을 생성한다.
- 파일을 생성할 때는 메모장을 이용하여 만들고, 아래 예시 코드를 각 Version에 맞게 작성한다.
- 나는 변경 후 버전 확인까지 하기 위해서 java -version 명령어를 추가했다.(빼도 상관없다.)

Java8 예시

```java
@echo off
set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_361
set Path=%JAVA_HOME%\bin;%Path%
echo Java 8 activated.
java -version
```

Java17 예시

```java
@echo off
set JAVA_HOME=C:\Program Files\Java\jdk-17
set Path=%JAVA_HOME%\bin;%Path%
echo Java 17 activated.
java -version
```

- 저장할 때에는 파일 형식을 ‘모든 파일’로 변경한 후 확장자 .bat를 입력해서 저장한다.

![Untitled 1](https://github.com/Byeon-MJ/TIL/assets/69300448/de7dbfe5-751e-414b-b82b-2f168eac89a1)

![Untitled 2](https://github.com/Byeon-MJ/TIL/assets/69300448/a8c06240-12b9-4561-b402-f0e6dd5dc7e4)

# 3. 환경 변수 설정

- JAVA_HOME 설정
    
    원하는 JDK 버전의 경로를 넣는다.
    
    나는 JDK17의 경로를 기본 JAVA_HOME 변수로 설정했다.

    ![Untitled 3](https://github.com/Byeon-MJ/TIL/assets/69300448/86c85301-5ea5-458f-9ca3-46218e068157)

- Path 추가
    1. 새로 만들기(N)를 선택하고 %JAVA_HOME%\bin 을 추가한다.
    2. .bat 파일 사용을 위해 scripts 폴더의 경로도 Path에 추가한다

    ![Untitled 4](https://github.com/Byeon-MJ/TIL/assets/69300448/75ab7928-a82f-487d-8e66-e68c4384b020)

# 4. JDK 변경 확인

- CMD 에서 .bat 파일명을 입력하면 Java version이 변경이 된다.

![Untitled 5](https://github.com/Byeon-MJ/TIL/assets/69300448/adb6d70d-b196-4224-9639-0472431069b1)


# Reference

[Java 버전 여러 개 전환하여 사용하는 법](https://almond0115.tistory.com/entry/Java-버전-여러-개-전환하여-사용하는-법)

[[자바, Java] 설치한 여러 JDK 간편하게 전환](https://computer-science-student.tistory.com/467)

[JDK 여러 버전 설치하여 사용하기](https://velog.io/@heyhighbyee/JDK-여러-버전-설치하여-사용하기)
