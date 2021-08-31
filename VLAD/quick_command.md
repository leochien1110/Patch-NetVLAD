This instruction lists the environment setup and script commands in simple way. The code is using TensorFlow to train(K-means) the data instead of CPU. 

You can run the script step-by-step to have better understanding about VLAD or run the all-in-one script to get the VLAD index file directly.
# Table of Contents
1. [Environment Setup](#environment-setup)
2. [Basic Operation](#basic-operation)
3. [Example Commands](#example-commands)

# Environment Setup
Install Aconda or Miniconda from the offical website.
### Build Conda Environment
> Change the `cudatoolit=________` for your CUDA version.
```bash
conda create -n openCV-Python python=3.8 psutil scikit-learn  matplotlib numpy scipy \
tensorflow-gpu cudatoolkit=10.2 opencv -c conda-forge
```
### Activate conda environment
```bash
conda activate openCV-Python
```
### Install TensorFlow package to accelerate K-Means training
```bash
pip install kmeanstf
```

# Basic Operation
## Build VLAD step-by-step
1. compute descriptors from a dataset. The supported descriptors are ORB, SIFT and SURF:
	```bash
	python describe.py --dataset dataset --descriptor descriptorName --output output
	```
2.  Construct a visual dictionary from the descriptors in path -d, with -w visual words:
	```bash
	python visualDictionary.py  -d descriptorPath -w numberOfVisualWords -o output
	```
    To train the dataset with CPU only, uncomment the following line in `VLAD.py`
    ```python
    est = KMeans(n_clusters=k,init='k-means++',tol=0.0001,verbose=1).fit(training)
    ```
    and comment out `KMeansTF`
    ```python
    est = KMeansTF(n_clusters=k,init='k-means++',tol=0.0001,verbose=1).fit(training)
    ```
3. Compute VLAD descriptors from the visual dictionary:
	```bash
	python vladDescriptors.py  -d dataset -dV visualDictionaryPath --descriptor descriptorName -o output
	```
4.  Make an index from VLAD descriptors using  a ball-tree DS:
	```bash
	python indexBallTree.py  -d VLADdescriptorPath -l leafSize -o output
	```
### 4-in-1: Wrap the 4 steps above
```bash
python describe.py --dataset dataset --descriptor descriptorName \
        --numberOfVisualWords numberOfVisualWords --leafSize leafSize \
        --output output
```
## Image Retrieval
Single Query Image:
```bash
python query.py  --query image --descriptor descriptor --index indexTree --retrieve retrieve
```
Read Query Images in a folder:
```bash
python VLAD_read.py  --query directory --descriptor descriptor --visualDictionary visualDictionary --index indexTree
```

# Example Commands
## ORB preset
### Build VLAD step-by-step
```bash
python describe.py -d dataset/index -n ORB -o descriptors/descriptorORB
```
```bash
python visualDictionary.py -d descriptors/descriptorORB.pickle  -w 16 -o visualDictionary/visualDictionary16ORB
```
```bash
python vladDescriptors.py  -d  dataset/index -dV visualDictionary/visualDictionary16ORB.pickle --descriptor ORB -o VLADdescriptors/VLAD_ORB_W16
```
```bash
python indexBallTree.py  -d VLADdescriptors/VLAD_ORB_W16.pickle -l 40 -o ballTreeIndexes/index_ORB_W16
```
### 4-in-1: Wrap the 4 steps above. This will only save the indexVLAD file.
```bash
python VLAD_database.py -d dataset/index -n ORB -w 16 -l 40 \
        -o ballTreeIndexes/index_ORB_W16
```
### Image Retrieval
```bash
python VLAD_read.py  -q dataset/query -d ORB -dV visualDictionary/visualDictionary16ORB.pickle -i ballTreeIndexes/index_ORB_W16.pickle
```
## SIFT preset
This descriptor could be fairly slow comparing to other two descriptors.
```bash
python describe.py --dataset dataset/index --descriptor SIFT --output descriptors/descriptorSIFT
```
```bash
python visualDictionary.py -d descriptors/descriptorSIFT.pickle  -w 16 -o visualDictionary/visualDictionary16SIFT
```
```bash
python vladDescriptors.py  -d  dataset/index -dV visualDictionary/visualDictionary16SIFT.pickle --descriptor SIFT -o VLADdescriptors/VLAD_SIFT_W16
```
```bash
python indexBallTree.py  -d VLADdescriptors/VLAD_SIFT_W16.pickle -l 40 -o ballTreeIndexes/index_SIFT_W16
```
```bash
python VLAD_read.py  -q dataset/query -d SIFT -dV visualDictionary/visualDictionary16SIFT.pickle -i ballTreeIndexes/index_SIFT_W16.pickle
```
## SURF preset
```bash
python describe.py --dataset dataset/index --descriptor SURF --output descriptors/descriptorSURF
```
```bash
python visualDictionary.py -d descriptors/descriptorSURF.pickle  -w 16 -o visualDictionary/visualDictionary16SURF
```
```bash
python vladDescriptors.py  -d  dataset/index -dV visualDictionary/visualDictionary16SURF.pickle --descriptor SURF -o VLADdescriptors/VLAD_SURF_W16
```
```bash
python indexBallTree.py  -d VLADdescriptors/VLAD_SURF_W16.pickle -l 40 -o ballTreeIndexes/index_SURF_W16
```
```bash
python VLAD_read.py  -q dataset/query -d SURF -dV visualDictionary/visualDictionary16SURF.pickle -i ballTreeIndexes/index_SURF_W16.pickle
```