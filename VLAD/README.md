# VLAD
## Synopsis

Python implementation of VLAD  for a CBIR system. 
Reference:

---
H. Jégou, F. Perronnin, M. Douze, J. Sánchez, P. Pérez and C. Schmid, "Aggregating Local Image Descriptors into Compact Codes," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, no. 9, pp. 1704-1716, Sept. 2012.
doi: 10.1109/TPAMI.2011.235

## Read this first!!
**This is a different independent repository from patch-netvlad.**
> `VLAD_database.py` and `VLAD_read.py` are modified from the original repository to build VLAD file and read query images from your dataset. Please check [Quick Command](./quick_command.md) for some preset commands including environment setup and basic operations.
>> Modified by Wen-Yu Chien(wen.yu.chien@nearearth.aero), 08/31/2021. For any build issue or bug, please check the [original repository](https://github.com/jorjasso/VLAD).

## Code Example

A query (-q queries/0706.3046-img-1-22.jpg) looking for the seven most similar images (-r 7) using ORB descriptors (-d ORB) and a visual vocabulary of 16 words (-dV visualDictionary/visualDictionary16ORB.pickle) and a ball-tree data structure as index (-i ballTreeIndexes/index_ORB_W16.pickle) is given by:

```python
python query.py  -q queries/0706.3046-img-1-22.jpg -r 7 -d ORB -dV visualDictionary/visualDictionary16ORB.pickle -i ballTreeIndexes/index_ORB_W16.pickle
```
You must compute the following first: Descriptors, Visual Dictionaries, ball-tree indexes and VLAD descriptors, see section "Computing VLAD features for a new dataset" below for details.


Another examples of queries:

SIFT
```python
python query.py  -q queries/0706.3046-img-1-22.jpg -r 7 -d SIFT -dV visualDictionary/visualDictionary16SIFT.pickle -i ballTreeIndexes/index_SIFT_W16.pickle

python query.py  -q queries/1403.3290-img-5-14.jpg -r 10 -d SIFT -dV visualDictionary/visualDictionary64SIFT.pickle -i ballTreeIndexes/index_SIFT_W64.pickle   

python query.py  -q queries/0801.2442-img-2-21.jpg -r 3 -d SIFT -dV visualDictionary/visualDictionary256SIFT.pickle -i ballTreeIndexes/index_SIFT_W256.pickle
```

SURF
```python
python query.py  -q queries/1409.1047-img-3-06.jpg -r 7 -d SURF -dV visualDictionary/visualDictionary256SURF.pickle -i ballTreeIndexes/index_SURF_W256.pickle

python query.py  -q queries/0903.1780-img-1-32.jpg -r 7 -d SURF -dV visualDictionary/visualDictionary256SURF.pickle -i ballTreeIndexes/index_SURF_W256.pickle

python query.py  -q queries/1409.1047-img-3-06.jpg -r 7 -d SURF -dV visualDictionary/visualDictionary16SURF.pickle -i ballTreeIndexes/index_SURF_W16.pickle
```


ORB
```python
python query.py  -q queries/0706.3046-img-1-22.jpg -r 7 -d ORB -dV visualDictionary/visualDictionary16ORB.pickle -i ballTreeIndexes/index_ORB_W16.pickle

python query.py  -q queries/1506.05863-img-3-21.jpg -r 7 -d ORB -dV visualDictionary/visualDictionary16ORB.pickle -i ballTreeIndexes/index_ORB_W16.pickle
```

## Computing VLAD features for a new dataset
Example VLAD with ORB descriptors with a visual dictionary with 2 visual words and an a ball tree as index. (Of course, 2 visual words is not useful, instead,  try 16, 32, 64, or 256 visual words)

Remark: Create folders: /ballTreeIndexes, /descriptors, /visualDictionary, /VLADdescriptors 

1. compute descriptors from a dataset. The supported descriptors are ORB, SIFT and SURF:
	```python
	python describe.py --dataset dataset --descriptor descriptorName --output output
	```
	*Example
	```python
	python describe.py --dataset dataset/index --descriptor ORB --output descriptors/descriptorORB
	```

2.  Construct a visual dictionary from the descriptors in path -d, with -w visual words:
	```python
	python visualDictionary.py  -d descriptorPath -w numberOfVisualWords -o output
	```
	*Example :
	```python
	python visualDictionary.py -d descriptors/descriptorORB.pickle  -w 16 -o visualDictionary/visualDictionary16ORB
	```

3. Compute VLAD descriptors from the visual dictionary:
	```python
	python vladDescriptors.py  -d dataset -dV visualDictionaryPath --descriptor descriptorName -o output
	```
	*Example :
	```python
	python vladDescriptors.py  -d  dataset/index -dV visualDictionary/visualDictionary16ORB.pickle --descriptor ORB -o VLADdescriptors/VLAD_ORB_W16
	```
	
4.  Make an index from VLAD descriptors using  a ball-tree DS:
	```python
	python indexBallTree.py  -d VLADdescriptorPath -l leafSize -o output
	```
	*Example :
	```python
	python indexBallTree.py  -d VLADdescriptors/VLAD_ORB_W16.pickle -l 40 -o ballTreeIndexes/index_ORB_W16
	```

5. Query:
	```python
	python query.py  --query image --descriptor descriptor --index indexTree --retrieve retrieve
	```
	*Example
	```python
	python query.py  -q dataset/query/0396.jpg -r 7 -d ORB -dV visualDictionary/visualDictionary16ORB.pickle -i ballTreeIndexes/index_ORB_W16.pickle
	```

6. Read Query images frame-by-frame:
	```python
	python VLAD_read.py  --query directory --descriptor descriptor --visualDictionary visualDictionary --index indexTree
	```
	*Example
	```python
	python VLAD_read.py  -q dataset/query -d ORB -dV visualDictionary/visualDictionary16ORB.pickle -i ballTreeIndexes/index_ORB_W16.pickle
	```

SIFT presets
```python
python describe.py --dataset dataset/index --descriptor SIFT --output descriptors/descriptorSIFT
```
```python
python visualDictionary.py -d descriptors/descriptorSIFT.pickle  -w 16 -o visualDictionary/visualDictionary16SIFT
```
```python
python vladDescriptors.py  -d  dataset/index -dV visualDictionary/visualDictionary16SIFT.pickle --descriptor SIFT -o VLADdescriptors/VLAD_SIFT_W16
```
```python
python indexBallTree.py  -d VLADdescriptors/VLAD_SIFT_W16.pickle -l 40 -o ballTreeIndexes/index_SIFT_W16
```
```python
python VLAD_read.py  -q dataset/query -d SIFT -dV visualDictionary/visualDictionary16SIFT.pickle -i ballTreeIndexes/index_SIFT_W16.pickle
```


## Installation

First install conda , then:

```python
conda create -n openCV-Python python=3.8 numpy scipy tensorflow-gpu cudatoolkit=10.2 \
	opencv scikit-learn  matplotlib-base -c conda-forge
```
```python
conda activate openCV-Python
```

## Contributor
jorge.jorjasso@gmail.com



