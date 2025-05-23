FROM  dustynv/ros:humble-ros-base-l4t-r36.2.0
# To make sure installing tzdata will not prompt questions
ARG DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y --no-install-recommends \
    # zsh \
    neovim tmux git htop curl wget \
    net-tools iputils-ping\
    # init certificate for curl
    ca-certificates \
    # For x11 forwarding tests (xeyes, xclock etc.)
    x11-apps \
    build-essential libboost-all-dev libeigen3-dev \
    # For pytorch
    libopenblas-dev \
    cmake sudo

ARG USERNAME=real
ARG USER_UID=1000
ARG USER_GID=1000


# Add the new user (userid should be consistent to the one in the host machine)
# Change the password to "real". You can use other password by changing the following line
RUN groupadd -g ${USER_GID} ${USERNAME} && \
    useradd ${USERNAME}  -u ${USER_UID} -g ${USER_GID} -m -p "$(openssl passwd -1 real)"
# Add sudo access (not recommended, but is hulepful for development)
RUN usermod -aG sudo ${USERNAME}
# Give ownership of the user's home directory to the new user
RUN chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}
USER ${USERNAME}
# Set the user's home directory as the working directory
# add to bashrc
RUN echo "export USERNAME=${USERNAME}" >> /home/${USERNAME}/.bashrc
WORKDIR /home/${USERNAME}


############### Development Tools ###############

# install and setup zsh (with oh-my-zsh and plugins)
# RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" 
# ARG PLUGIN_DIR=/home/${USERNAME}/.oh-my-zsh/custom/plugins/
# RUN git clone https://github.com/marlonrichert/zsh-autocomplete.git $PLUGIN_DIR/zsh-autocomplete && \
#     git clone https://github.com/zsh-users/zsh-autosuggestions $PLUGIN_DIR/zsh-autosuggestions  && \
#     git clone https://github.com/zsh-users/zsh-syntax-highlighting.git $PLUGIN_DIR/zsh-syntax-highlighting

# Set up zsh plugins and theme
# And some key bindings: ctrl-R to search in the history, ctlr-backspace to delete a word, etc.
# RUN mv /home/${USERNAME}/.zshrc /home/${USERNAME}/.zshrc.bak && \
#     echo 'export ZSH="$HOME/.oh-my-zsh"\nplugins=(\n  git\n  zsh-autosuggestions\n  zsh-autocomplete\n  zsh-syntax-highlighting\n  themes\n)\n' >> /home/${USERNAME}/.zshrc && \
#     echo 'ZSH_THEME="eastwood"\nsource $ZSH/oh-my-zsh.sh' >> /home/${USERNAME}/.zshrc && \
#     echo 'bindkey -M menuselect "\\r" .accept-line' >> /home/${USERNAME}/.zshrc && \
#     echo 'bindkey -M menuselect -s "^R" "^_^_^R" "^S" "^_^_^S"' >> /home/${USERNAME}/.zshrc && \
#     echo 'bindkey -M menuselect "\\e[D" .backward-char "\\eOD" .backward-char "\\e[C" .forward-char "\\eOC" .forward-char' >> /home/${USERNAME}/.zshrc && \
#     echo 'bindkey '^H' backward-kill-word' >> /home/${USERNAME}/.zshrc

# Setup python env with Python 3.10 by default
RUN ARCH=$(uname -m) && wget https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Miniforge3-24.3.0-0-Linux-${ARCH}.sh && \
    bash Miniforge3-24.3.0-0-Linux-${ARCH}.sh -b -p /home/${USERNAME}/miniforge3 && \
    rm Miniforge3-24.3.0-0-Linux-${ARCH}.sh && \
    /home/${USERNAME}/miniforge3/bin/conda init bash && \
    /home/${USERNAME}/miniforge3/bin/mamba init bash

# pre-install vscode server and helpful plugins. This helps to reduce the time to edit the dockerfile and reopen the container.
RUN git clone https://gist.github.com/0a16b6645ab7921b0910603dfb85e4fb.git /home/${USERNAME}/vscode-install && \
    chmod +x /home/${USERNAME}/vscode-install/download-vs-code-server.sh && \
    /home/${USERNAME}/vscode-install/download-vs-code-server.sh linux
ENV PATH=/home/${USERNAME}/.vscode-server/bin/default_version/bin:$PATH
RUN code-server --install-extension ms-python.python && \
    code-server --install-extension mhutchie.git-graph && \
    code-server --install-extension eamodio.gitlens && \
    code-server --install-extension github.copilot && \
    code-server --install-extension kevinrose.vsc-python-indent && \
    code-server --install-extension streetsidesoftware.code-spell-checker && \
    code-server --install-extension ms-python.black-formatter

# Enable using `code` command in terminal to open file or attach new window to a folder
RUN echo "export PATH=/home/${USERNAME}/.vscode-server/bin/default_version/bin/remote-cli:\$PATH" >> /home/${USERNAME}/.bashrc


# Install pytorch. 
# It does not have access to cuda kernels as the Jetpack version on Unitree jetson is usually lower than 6.0.
# In our project, the whole-body controller only uses CPU and diffusion policy is run on a separate PC
#   thus cuda compute is not required in our project
RUN /home/${USERNAME}/miniforge3/bin/pip install torch \
--extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/
# To get torch working on earlier Jetpack version, you can try python3.8 versions in https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
#   Only Python 3.8 is supported on versions earlier than Jetpack 6.0 

RUN /home/${USERNAME}/miniforge3/bin/pip install numpy

# Install ROS2
USER root
RUN apt-get update && \
    apt-get install -y \
    ros-humble-rmw-cyclonedds-cpp ros-humble-rosidl-generator-dds-idl \
    ros-humble-geometry-msgs ros-humble-ament-lint-auto ros-humble-rosidl-default-generators \
    ros-humble-rviz2 \
    ros-humble-dynamixel-sdk \
    ros-humble-ros2*
USER ${USERNAME}

# Install unitree ros
RUN git clone https://github.com/unitreerobotics/unitree_ros2.git && \
    cd unitree_ros2/cyclonedds_ws/src && \
    git clone https://github.com/ros2/rmw_cyclonedds -b humble && \
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x && \
    cd .. && \
    colcon build --packages-select cyclonedds

RUN cd unitree_ros2/cyclonedds_ws && \
    /home/${USERNAME}/miniforge3/bin/pip install empy==3.3.2 catkin_pkg pyparsing lark && \
    /bin/bash -c "cd ~/unitree_ros2/cyclonedds_ws && source /opt/ros/humble/setup.bash && colcon build"

RUN sed -i "s/\\\\\\\\/\\\\/g" /home/${USERNAME}/.bashrc && \
    sed -i "s/\\\\n/\\n/g" /home/${USERNAME}/.bashrc


# Install the customized unitree sdk for 
# 1) a python module to calculate CRC code for ros2 packages
# 2) a script to disable sports mode of Go2
RUN git clone https://github.com/yihuai-gao/unitree_sdk2.git && \
    cd unitree_sdk2 && \
    git checkout 92506a409843c78be358d761d19aa254053a045f && \
    mkdir build && cd build && \
    cmake .. && make -j$(($(nproc)-1)) && \
    /home/${USERNAME}/miniforge3/bin/pip install pybind11 && \
    cd ../python && mkdir build && cd build &&  \
    cmake .. -DPYBIND_PATH=$(/home/${USERNAME}/miniforge3/bin/pip show pybind11 | grep Location | cut -d' ' -f2) && \
    make -j$(($(nproc)-1)) && \
    echo "export PYTHONPATH=\$PYTHONPATH:/home/${USERNAME}/unitree_sdk2/python" >> /home/${USERNAME}/.bashrc

USER root
RUN cd /home/${USERNAME}/unitree_sdk2 && ./install.sh

USER ${USERNAME}

RUN echo "source /opt/ros/humble/install/setup.bash" >> /home/${USERNAME}/.bashrc && \
    echo "source /home/\${USERNAME}/unitree_ros2/cyclonedds_ws/install/setup.bash" >> /home/${USERNAME}/.bashrc && \
    echo "export CYCLONEDDS_URI=/home/\${USERNAME}/unitree_ros2/cyclonedds_ws/src/cyclonedds.xml" >> /home/${USERNAME}/.bashrc && \
    sed -i "s/enp2s0/eth0/g" /home/${USERNAME}/unitree_ros2/cyclonedds_ws/src/cyclonedds.xml

RUN /home/${USERNAME}/miniforge3/bin/pip install transforms3d hydra-core filterpy numba omegaconf gitpython zarr rich python-socketio eventlet
RUN /home/${USERNAME}/miniforge3/bin/mamba install -y matplotlib seaborn


USER root

RUN apt-get update && \
    #zsh\
    apt-get install -y git curl git vim net-tools wget htop \ 
    x11-apps neofetch build-essential libboost-all-dev libeigen3-dev && \
    apt-get install libtinyxml2-dev libcppunit-dev liborocos-kdl-dev liburdfdom-dev iproute2 -y
RUN ln -s /usr/include/eigen3/Eigen /usr/local/include/Eigen

RUN git clone https://github.com/ros/kdl_parser.git && \
    cd kdl_parser && \
    git checkout 74d4ee3bc6938de8ae40a700997baef06114ea1b && \
    cd kdl_parser && \
    sed -i 's/kdl_parser\/visibility_control.hpp/visibility_control.hpp/g' include/kdl_parser/kdl_parser.hpp && \
    mkdir build && cd build && \
    cmake .. && make -j$(($(nproc)-1)) && make install


USER ${USERNAME}
RUN echo "source /home/\${USERNAME}/real-wbc/ros2/install/setup.bash" >> /home/${USERNAME}/.bashrc

USER root

# Setup spacenav and rosbag2 storage
RUN apt-get update && \
    apt-get install -y ros-humble-rosbag2-storage-mcap && \
    apt-get install -y libspnav-dev spacenavd && \
    apt-get install apt-utils -y && \
    apt-get install -f

# Add supports for Interbotix WidowX 250s robot arm
ENV ROBOT_MODEL="wx250s"
RUN apt install udev -y
RUN curl 'https://raw.githubusercontent.com/alvarobelmontebaeza/interbotix_ros_manipulators/refs/heads/main/interbotix_ros_xsarms/install/rpi4/xsarm_rpi4_install.sh' > xsarm_rpi4_install.sh
RUN chmod +x xsarm_rpi4_install.sh
RUN ./xsarm_rpi4_install.sh -d humble -n -p "/home/${USERNAME}/interbotix_ws"

USER ${USERNAME}

# Install modern robotics package used for FK and IK
RUN /home/${USERNAME}/miniforge3/bin/pip install modern_robotics

RUN echo "source /opt/ros/humble/setup.bash" >> /home/${USERNAME}/.bashrc && \
    echo "source /home/\${USERNAME}/interbotix_ws/install/setup.bash" >> /home/${USERNAME}/.bashrc
