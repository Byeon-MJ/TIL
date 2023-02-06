# [MySQL] 내가 보려고 정리한 MySQL 명령어

### 서비스(서버) 시작하기

```python
net start mysql
```

### 서비스(서버) 종료하기

```python
net stop mysql
```

### 접속하기

서비스 구동이 안되어 있으면 아래 에러 발생

![Untitled](%5BMySQL%5D%20%E1%84%82%E1%85%A2%E1%84%80%E1%85%A1%20%E1%84%87%E1%85%A9%E1%84%85%E1%85%A7%E1%84%80%E1%85%A9%20%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5%E1%84%92%E1%85%A1%E1%86%AB%20MySQL%20%E1%84%86%E1%85%A7%E1%86%BC%E1%84%85%E1%85%A7%E1%86%BC%E1%84%8B%E1%85%A5%201f48c6a688dd421da9478b1a83a94722/Untitled.png)

```python
mysql -u root -p → password

mysql -u [user name] -p
```

### MySQL 종료하기

```python
EXIT
```

### 상태 확인해보기

```sql
STATUS;
```

### 데이터베이스 조회, 생성, 삭제, 사용

```sql
SHOW DATABASES;

CREATE DATABASE DBname;

DROP DATABASE DBname;

USE DBname;
```

### 비밀번호 변경(MySQL 접속 후 사용)

```sql
ALTER USER 'root'@'localhost' IDENTIFIED WITH caching_sha2_password BY '1234';
```

### 비밀번호 변경 (분실시)

```sql
mysql.server stop

mysql.server start --skip-grant-tables

mysql -u root

USE mysql;

UPDATE user SET authentication_string=null WHERE User='root';

FLUSH PRIVILEGES;

EXIT

mysql -u root

***ALTER USER 'root'@'localhost' IDENTIFIED WITH caching_sha2_password BY '1234';***
```