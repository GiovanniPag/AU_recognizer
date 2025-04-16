#!/bin/bash
echo "Installing mamba"
conda install mamba -n base -c conda-forge
if ! command -v mamba &> /dev/null
then
    echo "mamba could not be found. Please install mamba before running this script"
    exit
fi
echo "Creating conda environment"
mamba create -n au_env python=3.11
eval "$(conda shell.bash hook)" # make sure conda works in the shell script
conda activate au_env
if echo $CONDA_PREFIX | grep au_env
then
    echo "Conda environment successfully activated"
else
    echo "Conda environment not activated. Probably it was not created successfully for some reason. Please activate the conda environment before running this script"
    exit
fi
echo "Installing conda packages"
mamba env update -n au_env --file conda-environment_py311_cu121.yml
echo "Making sure Pytorch3D installed correctly"
pip install git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8
echo "Installation finished"
