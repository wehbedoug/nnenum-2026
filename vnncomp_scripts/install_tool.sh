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



#### SDW: Gurobi License setup, copied from Vibecheck

# --- Gurobi license link printout (always runs) -------------------------------
# Fetches grbprobe (pip gurobipy ships none) and prints the node-locked
# HOSTID/HOSTNAME/USERNAME/CORES plus a Gurobi keyserver URL. For the license-
# provisioning submission: open that URL from a university network to mint the
# .lic, paste it into post_install_tool.sh, and upload that as the post-install
# script in the VNNCOMP form. Harmless on non-Gurobi runs (just the 62 MB fetch
# + printout); the hostid-mismatch line below only warns when the licensing ENI
# (MAC 02:f3:7e:22:18:25 -> hostid 7e221825) is not the attached interface.
echo "==> Gurobi license link"
GRB_BIN=/tmp/gurobi1302/linux64/bin
if [ ! -x "$GRB_BIN/grbprobe" ]; then
	echo "Fetching grbprobe..."
	curl -sL -o /tmp/gurobi1302.tar.gz https://packages.gurobi.com/13.0/gurobi13.0.2_linux64.tar.gz \
		&& tar xzf /tmp/gurobi1302.tar.gz -C /tmp \
			gurobi1302/linux64/bin/grbprobe gurobi1302/linux64/bin/gurobi_cl \
		|| echo "WARNING: grbprobe fetch failed; skipping license link"
fi

grbprobe_output=$("$GRB_BIN/grbprobe" 2>/dev/null || true)
echo "$grbprobe_output"

HOSTNAME=$(echo $grbprobe_output | grep -Po "(?<=HOSTNAME=)(.*?)(?= )" || true)
HOSTID=$(echo $grbprobe_output | grep -Po "(?<=HOSTID=)(.*?)(?= )" || true)
USERNAME=$(echo $grbprobe_output | grep -Po "(?<=USERNAME=)(.*?)(?= )" || true)
CORES=$(echo $grbprobe_output | grep -Po "(?<=CORES=)(.*?)(?= )" || true)
LOCALDATE=$(date -u +%F)

# The 2026 VNNCOMP vibecheck eval uses a fixed licensing ENI (MAC
# 02:f3:7e:22:18:25) -> hostid must be 7e221825.
EXPECTED_HOSTID=7e221825
if [ "$HOSTID" != "$EXPECTED_HOSTID" ]; then
	echo "WARNING: HOSTID=$HOSTID does not match the expected ENI hostid ($EXPECTED_HOSTID) for the 2026 VNNCOMP vibecheck eval -- the license in post_install_tool.sh will NOT match this machine."
fi

# SDW: insert academic license key here
KEY=to-be-filled
probe_url="https://portal.gurobi.com/keyserver?id=${KEY}&hostname=${HOSTNAME}&hostid=${HOSTID}&username=${USERNAME}&os=linux&localdate=${LOCALDATE}&version=13&cores=${CORES}"

echo ""
echo "Node-locked Gurobi license setup (one-time; reused on every run via the fixed ENI):"
echo "  1. Get an academic Named-User license KEY (node-locked, NOT WLS) at:"
echo "       https://portal.gurobi.com/iam/licenses/request/?type=academic"
echo "  2. In the URL below, replace  id=to-be-filled  with your KEY."
echo "  3. From a UNIVERSITY network (campus, or a laptop on the campus VPN), fetch"
echo "     it with curl -- a browser does NOT work (it redirects to a login page):"
echo "       curl \"<the URL, with your KEY in id=>\""
echo "  4. curl returns the license text (TYPE=.../VERSION=13/HOSTID=7e221825/KEY=.../CKEY=...)."
echo "  5. Paste that text into vnncomp_scripts/post_install_tool.sh between the EOF"
echo "     markers, then upload that script as the post-install script in the web form."
echo ""
echo "  License-request URL (put your KEY in id=):"
echo "$probe_url"
