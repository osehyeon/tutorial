# 베이스 이미지 선택 (Ubuntu 20.04 for ARMv8)
FROM arm64v8/ubuntu:20.04

# 필요한 패키지 업데이트 및 설치
RUN apt-get update && apt-get install -y openssh-server sudo

# SSH 서버 설정
RUN mkdir /var/run/sshd

# 사용자 'ubuntu' 생성 및 설정
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 ubuntu
RUN echo 'ubuntu:ubuntu' | chpasswd

# 'ubuntu' 사용자에게 sudo 권한 부여
RUN echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# SSH 설정 변경
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH 포트 열기
EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]