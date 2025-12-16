# ROS_GO2

# WSL2 + Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic + RL 환경 세팅 로그

## 0. 환경 정보

- Host: Windows 10/11
- WSL2
- Linux 배포판: **Ubuntu 24.04.3 LTS (noble)**
- 커널: `6.6.87.2-microsoft-standard-WSL2`
- ROS 2: **Jazzy**
- Gazebo: **Gazebo Harmonic**
- Python RL 환경:
  - Python 3.12
  - `gymnasium`, `stable-baselines3`, `torch`, `numpy`

---

## 1. 기본 패키지 & 시스템 업데이트

```bash
sudo apt update
sudo apt upgrade -y

# 편의를 위한 기본 도구
sudo apt install -y curl gnupg lsb-release


## 2. Jazzy Desktop 설치
공식 ROS2 저장소 키 및 레포지토리 추가 후 설치

# 키 저장 디렉토리 생성
sudo mkdir -p /etc/apt/keyrings

# ROS 2 GPG 키 추가
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  | sudo tee /etc/apt/keyrings/ros-archive-keyring.gpg > /dev/null

# ROS 2 APT 레포지토리 등록
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
| sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 패키지 목록 갱신 및 ROS 2 설치
sudo apt update
sudo apt install -y ros-jazzy-desktop

# ROS 2 자동 로드를 위해 ~/.bashrc에 추가
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc

# 확인
ros2 -h | head -n 5
rosdep --version
colcon --help | head -n 3

## 3. Gazebo Harmonic 설치
OSRF(Gazebo) 공식 레포지토리 추가 후 Gazebo Harmonic 설치

# GPG 키 다운로드
sudo curl https://packages.osrfoundation.org/gazebo.gpg \
  --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg

# Gazebo APT 레포지토리 등록
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
| sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# 패키지 목록 갱신 및 Gazebo Harmonic 설치
sudo apt update
sudo apt install -y gz-harmonic

# 설치확인:
which gz
gz --commands      # 기본적으로 sdf 유틸리티 표시
ls /usr/share/gz   # harmonic 관련 설정/리소스 확인

## 4. X11 GUI 연동 (Windows + VcXsrv)
WSL2에서 Gazebo GUI 등을 띄우기 위해 Windows 쪽에 X 서버를 설치하고 DISPLAY에 연결

# 4.1 Windows에 VcXsrv 설치
winget search vcxsrv #PowerShell 또는 CMD 에서
winget install marha.VcXsrv

# 4.2 WSL2에서 DISPLAY 설정
# Windows 호스트 IP 자동 추출
WINIP=$(ip route | awk '/default/ {for(i=1;i<=NF;i++) if($i=="via"){print $(i+1); exit}}')

# DISPLAY 환경변수 설정
export DISPLAY=$WINIP:0
export LIBGL_ALWAYS_INDIRECT=1   # 필요시

# 테스트용 X11 앱 설치 및 확인
sudo apt install -y x11-apps
xclock    # 시계 창이 Windows 쪽에 뜨면 OK

## 5. Python RL 전용 가상환경 생성

# 5.1 venv 및 pip 설치
sudo apt update
sudo apt install -y python3.12-venv python3-pip

# 5.2. 가상환경 생성 및 활성화
# 홈 디렉토리에 RL 전용 환경 생성
python3 -m venv ~/RL_Robots

# 활성화
source ~/RL_Robots/bin/activate

# 5.3. RL 관련 패키지 설치
# 최신 pip으로 업그레이드
pip install --upgrade pip

# 기본 RL 스택 설치
pip install numpy gymnasium stable-baselines3

# 설치확인
python -c "import numpy, gymnasium as gym, stable_baselines3, torch; \
print('OK'); \
print('numpy', numpy.__version__); \
print('gym', gym.__version__); \
print('sb3', stable_baselines3.__version__); \
print('torch', torch.__version__)"

# 예시 출력
OK
numpy 2.3.5
gym 1.2.2
sb3 2.7.1
torch 2.9.1+cu128

## 6.1 CartPole PPO 테스트 스크립트
# 6.1 테스트 스크립트 작성
cd ~/ws_moveit
nano test_cartpole.py

import gymnasium as gym
from stable_baselines3 import PPO


def main():
    # CartPole 환경 생성
    env = gym.make("CartPole-v1")

    # PPO 모델 초기화 및 학습
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    # 학습된 정책으로 200 스텝 테스트
    obs, _ = env.reset()
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            obs, _ = env.reset()

    env.close()
    print("CartPole 테스트 완료!")


if __name__ == "__main__":
    main()

# 가상환경 활성화 상태여야 함
source ~/RL_Robots/bin/activate

cd ~/ws_moveit
python test_cartpole.py
