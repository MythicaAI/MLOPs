###################################################################
#This script is intended for manual runs based on the target system 
# DO NOT RUN IT UNATTENDED


#############################################
# CUDA Prereqs
#

#Detect Linux version
source /etc/os-release
clean_version_id=$(echo "$VERSION_ID" | tr -cd '[:digit:]')
VERSION="${ID,,}$clean_version_id"

#Install cuda keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/$VERSION/x86_64/cuda-keyring_1.1-1_all.deb

sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit git

# TO check cuda version
/usr/local/cuda/bin/nvcc --version

# To remove Cuda
sudo apt-get remove cuda-toolkit
sudo apt-get autoremove 
sudo apt-get autoclean



#############################################
# Conda env 

#Install Miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash

#Create and activate env
conda create -n oneformer python=3.10
conda activate oneformer

conda deactivate
conda env remove -n oneformer


#############################################
# PIP requirements

pip install torch torchvision transformers labelbox geojson pygeotile imagesize scipy
pip install git+https://github.com/cocodataset/panopticapi.git                 
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install labelbox[data] --upgrade

#############################################
# Clone the repo

sudo apt-get install git
git clone https://github.com/MythicaAI/MLOPs.git
cd MLOPs
git switch oneformer
# link scripts/python/training is in the PYTHONPATH (or echo it in to ~/.bash_profile for persistence)
export PYTHONPATH=`pwd`/scripts/python/mythica:$PYTHONPATH

#############################################
# To be executed inside a Python shell!

python

#Imports
import oneformer as of
import os

#model source and target
model_source='shi-labs/oneformer_coco_swin_large'
model_target='mythica'

#labelbox params
lb_project_id='#GENERATE_AT_LABELBOX'
lb_api_key="#GENERATE_AT_LABELBOX"

#cache locations for labelbox and models
cache=os.path.join(os.path.expanduser('~'),'oneformer')
cache_lb=os.path.join(cache,'labelbox')
cache_model=os.path.join(cache,'model')

#inference params
input_url='https://storage.labelbox.com/clplo62e508hv07x6el3c9he8%2F82435119-caeb-0681-7aa1-5f95531cdcc6-fffa686a-828a-43f9-ae72-7137bda06743.png?Expires=1707867330842&KeyName=labelbox-assets-key-3&Signature=MoCsu3CFeKcF1eySZa1BLJT3F1A'
output_dir=os.path.join(cache,'output')


#Cache Label Box Data
of.cache(lb_api_key,lb_project_id,cache_lb)

#Train teh model
of.train(model_source,model_target,cache_model,cache_lb)

#Use the source model
of.infer(model_source,cache_model,input_url,output_dir)
#Use our trained model
of.infer(os.path.join(cache_model,model_target),cache_model,input_url,output_dir)
