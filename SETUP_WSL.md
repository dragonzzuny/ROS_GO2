# WSL에서 RL Dispatch MVP 설정하기

현재 WSL Ubuntu 환경에 Python이 설치되어 있지 않습니다. 다음 단계를 따라 설정하세요.

## 방법 1: Windows PowerShell에서 WSL Python 설치

Windows PowerShell을 **관리자 권한**으로 실행하고:

```powershell
# WSL Ubuntu에 접속
wsl

# 또는 직접 명령 실행
wsl -e bash -c "apt update && apt install -y python3 python3-pip python3-venv"
```

## 방법 2: Ubuntu 터미널에서 직접 설치

1. **Ubuntu 앱** 실행 (Windows Store에서 설치한 Ubuntu)

2. 다음 명령어 실행:

```bash
# 시스템 업데이트
sudo apt update

# Python 설치
sudo apt install -y python3 python3-pip python3-venv

# 설치 확인
python3 --version
pip3 --version
```

## 방법 3: Windows에서 직접 사용 (권장)

WSL 대신 Windows에서 직접 Python을 사용하는 것이 더 간단할 수 있습니다:

### 1. Python 설치

https://www.python.org/downloads/ 에서 Python 3.8 이상 다운로드 및 설치

**중요**: 설치 시 "Add Python to PATH" 체크!

### 2. PowerShell에서 프로젝트로 이동

```powershell
cd \\wsl.localhost\Ubuntu-22.04\home\yjp\rl_dispatch_mvp

# 또는 로컬로 복사
xcopy \\wsl.localhost\Ubuntu-22.04\home\yjp\rl_dispatch_mvp C:\rl_dispatch_mvp /E /I
cd C:\rl_dispatch_mvp
```

### 3. 가상 환경 생성 및 활성화

```powershell
# 가상 환경 생성
python -m venv venv

# 활성화
.\venv\Scripts\activate

# 설치
pip install -e .
```

### 4. 테스트

```powershell
# 데모 실행
python scripts/demo.py

# 또는 Python에서 직접
python -c "from rl_dispatch.env import PatrolEnv; print('Success!')"
```

## 빠른 테스트 (Python 설치 후)

```bash
# WSL에서
cd ~/rl_dispatch_mvp
python3 -m pip install --user -e .
python3 scripts/demo.py
```

또는

```powershell
# Windows PowerShell에서
cd C:\rl_dispatch_mvp
python -m pip install -e .
python scripts/demo.py
```

## 문제 해결

### "pip: command not found"
```bash
python3 -m pip install --upgrade pip
```

### "Permission denied"
```bash
python3 -m pip install --user -e .
```

### WSL에서 계속 문제가 있다면
Windows에서 직접 사용하는 것을 권장합니다 (방법 3).

## 다음 단계

설치가 완료되면:

1. **빠른 테스트**: `python scripts/demo.py`
2. **학습 시작**: `python scripts/train.py`
3. **문서 읽기**: `README.md`, `QUICK_START.md`

---

도움이 필요하면 INSTALL.md를 참조하세요.
