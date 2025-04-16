
<h1 align="center">AU Recognizer: GUI for 3DMM mesh fitting and Mesh AU Tagging</h1>
<p align="center">





<p align="center"> 
<img src="AU_recognizer/var/images/splash.png">
</p>
<p align="center">home screen of the program.<p align="center">


AU Recognizer currently uses EMOCA v2 as a 3dmm Model to take in-the-wild images as input and reconstructs a 3D face mesh, then with a neutral face from the same subject gives the option to see the difference between the two mesh by taking advantage of the FLAME topology, and if the Action Unit coding displayed by the faces are provided it can for each point of the face tag it with the action units that can move it. EMOCA official project page is [here](https://emoca.is.tue.mpg.de/index.html).
 
## Installation 

### Dependencies

1) Install [conda](https://docs.conda.io/en/latest/miniconda.html)


2) Clone this repo
### Short version
1) Run the installation script:
```bash
bash install_au_env_311.sh
```

2) Pull the models of Emoca, needed to run the fitting, you will need to accept the license terms at [FLAME](https://flame.is.tue.mpg.de) and [EMOCA]( https://emoca.is.tue.mpg.de):

```bash
bash download_assets.sh
```

If this ran without any errors, you now have a functioning conda environment with all the necessary packages to [run the code](#usage). If you had issues with the installation script, go through the [long version](#long-version) of the installation and see what went wrong. Certain packages (especially for CUDA, PyTorch and PyTorch3D) may cause compatibility issues.

### Long version

1) Set up a conda environment with the provided conda file.

You can use [mamba](https://github.com/mamba-org/mamba) to create a conda environment (strongly recommended):

```bash
mamba env create python=3.11 --file conda-environment_py311_cu121.yml
```

but you can also use plain conda if you want (but it will be slower): 
```bash
conda env create python=3.11 --file conda-environment_py311_cu121.yml
```

In case the specified pytorch version somehow did not install, try again manually: 
```bash
mamba install pytorch==2.4.0 torchvision torchaudio cudatoolkit=12.1 -c pytorch
```

2) Activate the environment: 
```bash 
conda activate au_env
```

3) sometimes cython may glitch, so install it separately: 
```bash 
pip install Cython==0.29
```

4) Verify that previous step correctly installed Pytorch3D

For some people the compilation fails during requirements install and works after. Try running the following separately:

```bash
pip install git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8
```

Pytorch3D installation (which is part of the installation script and needed by EMOCA) can unfortunately be tricky and machine specific. for pytorch3D, pytorch needs to be compiled in a CUDA version supported by your PC, you can check it with `nvcc --version`. If it fails to compile, you can try to find another way to install Pytorch3D.

5) Pull the models of Emoca, needed to run the fitting, you will need to accept the license terms at [FLAME](https://flame.is.tue.mpg.de) and [EMOCA]( https://emoca.is.tue.mpg.de):

```bash
bash download_assets.sh
```

## Usage

0) Activate the environment: 
```bash
mamba activate au_env
```

1) For running Au_recognizer, run `main.py` 
```bash
python main.py
```


## Structure 
This repo has two subpackages. `gdl` and `gdl_apps` 

### GDL
`gdl` is a library full of research code. Some things are OK organized, some things are badly organized. It includes but is not limited to the following: 

- `models` is a module with (larger) deep learning modules (pytorch based) 
- `layers` contains individual deep learning layers 
- `datasets` contains base classes and their implementations for various datasets I had to use at some points. It's mostly image-based datasets with various forms of GT if any
- `utils` - various tools

Emoca is heavily based on PyTorch and Pytorch Lightning.


## License
This code and model are **available for non-commercial scientific research purposes** as defined in the [LICENSE](https://emoca.is.tue.mpg.de/license.html) file. By downloading and using the code and model you agree to the terms of this license. 

## Acknowledgements 
There are many people who deserve to get credited. These include but are not limited to: 
Yao Feng and Haiwen Feng and their original implementation of [DECA](https://github.com/YadiraF/DECA).
Antoine Toisoul and colleagues for [EmoNet](https://github.com/face-analysis/emonet).
