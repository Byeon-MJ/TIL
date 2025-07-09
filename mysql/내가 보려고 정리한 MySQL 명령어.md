# 내가 보려고 정리한 MySQL 명령어

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

![Untitled](https://user-images.githubusercontent.com/69300448/216914462-8a0979e3-3043-40a5-808e-a0fffa31b4d4.png)

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
