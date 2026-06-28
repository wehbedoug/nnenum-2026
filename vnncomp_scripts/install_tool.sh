#!/bin/bash
# install_tool.sh script for VNNCOMP for nnenum
# Stanley Bak

TOOL_NAME=nnenum
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

echo "Installing $TOOL_NAME"
DIR=$(dirname $(dirname $(realpath $0)))

#######################pip#########################
# apt-get update &&
# apt-get install -y python3.8 python3-pip &&
# apt-get install -y psmisc && # for killall, used in prepare_instance.sh script
# pip3 install -r "$DIR/requirements.txt"
################################################


#######################pipenv#########################
# sudo apt-get update

# sudo apt-get remove -y python2.7 python3.6
# sudo apt-get autoremove -y
# sudo apt-get install -y python3.8 python3.8-dev gfortran python3-pip bc

# python3.8 -m pip install pip
# sudo -H python3.8 -m pip install -U pipenv
# pipenv install python 3.8
# pipenv install -r "$DIR/requirements.txt"
# pipenv_python=`pipenv run which python`

# # Gurobi
# cd ~/
# wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz
# tar -xzvf gurobi9.1.2_linux64.tar.gz
# rm gurobi9.1.2_linux64.tar.gz
# sudo mv gurobi912/ /opt/
# # mv gurobi912/ /opt/
# cd /opt/gurobi912/linux64/
# $pipenv_python setup.py install

# echo "grbprobe"
# cd $DIR
# pipenv run grbprobe

# echo "pipenv --venv"
# pipenv --venv
#######################pipenv#########################


#######################conda#########################
# conda_path = ${HOME}/anaconda3/bin
# py_pip_path = ${HOME}/anaconda3/bin

# conda_path = ${HOME}/miniconda/bin
# py_pip_path = ${HOME}/miniconda/envs/nnenumenv/bin # path for python, pip, grbprobe
# download and install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
# wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O anaconda.sh
sh miniconda.sh -b -p ${HOME}/miniconda
# sh miniconda.sh -b
# sh anaconda.sh -b
# echo 'export PATH=${PATH}:'${HOME}'/miniconda/bin' >> ~/.profile
echo 'export PATH=${PATH}:'${DIR}'/miniconda/bin' >> ~/.profile
# echo 'export PATH=${PATH}:'${HOME}'/anaconda3/bin' >> ~/.profile
# # echo "alias py38=\"conda activate nnenumenv\"" >> ${HOME}/.profile
echo "conda activate nnenumenv" >> ${HOME}/.profile
# export PATH=${PATH}:$HOME/miniconda/bin
export PATH=${PATH}:$DIR/miniconda/bin

# see all the process
# ps aux
# create conda environment
# ${HOME}/miniconda/bin/conda env create -f ${DIR}/environment.yml

# 1. Make Conda auto-accept the ToS
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=true

# 2. Ensure the plugin that understands the env-var is present
${HOME}/miniconda/bin/conda install --name base conda-anaconda-tos
# main repository
${HOME}/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# R repository
${HOME}/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ${HOME}/anaconda3/bin/conda create --yes --name nnenumenv python=3.8
${HOME}/miniconda/bin/conda create --yes --name nnenumenv python=3.8
# ${HOME}/miniconda/bin/conda activate nnenumenv

# ${HOME}/anaconda3/envs/nnenumenv/bin/pip install -r "$DIR/requirements.txt"
${HOME}/miniconda/envs/nnenumenv/bin/pip install -r "$DIR/requirements.txt"
# ${HOME}/anaconda3/envs/nnenumenv/bin/pip install -U --no-deps git+https://github.com/dlshriver/DNNV.git@4d4b124bd739b4ddc8c68fed1af3f85b90386155#egg=dnnv
${HOME}/miniconda/envs/nnenumenv/bin/pip install -U --no-deps git+https://github.com/dlshriver/DNNV.git@4d4b124bd739b4ddc8c68fed1af3f85b90386155#egg=dnnv

# ${HOME}/anaconda3/bin/conda install -y -c gurobi gurobi
# ${HOME}/anaconda3/bin/conda install --yes -n nnenumenv -c gurobi gurobi
${HOME}/miniconda/bin/conda install --yes -n nnenumenv -c gurobi gurobi

# Run grbprobe for activating gurobi later.
# ${HOME}/anaconda3/envs/nnenumenv/bin/grbprobe
${HOME}/miniconda/envs/nnenumenv/bin/grbprobe