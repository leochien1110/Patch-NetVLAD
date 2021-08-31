This instruction lists the environment setup and script commands in simple way.
# Table of Contents
1. [Environment Setup](#environment-setup)
2. [Basic Operation](#basic-operation)
3. [Feature Matching Webcam Stream](#feature-matching-from-webcam-stream)
3. [Feature Matching from dataset images](#feature-matching-from-dataset-images)
4. [FAQ](#faq)

# Environment Setup
Install Aconda or Miniconda from the offical website.
### Build Conda Environment
> Change the `cudatoolit=________` for your CUDA version. Use `nvcc -V` to check.

`conda create -n patchnetvlad python 3.8 numpy pytorch-gpu torchvision \
cudatoolkit 10.2 natsort tqdm opencv pillow scikit-learn faiss matplotlib-base \
-c conda-forge`

### Activate conda environment

`conda activate patchnetvlad`

# Basic Operation
## Configuration files
The configuration must be the same for feature extraction and feature matching.
1. Performance: the most robust recognition but slow for real-time process
2. performance: Fast for both setting up database and real-time process. Less accurate.
3. performance: Use less space in the drive and fastest. The most inaccurate.

> For performance configuration, make sure to have enough disk space or the code might stop without enough space. For Pittsburgh30k, it takes about 500GB.
## Dataset and Path Setup
Setup your own dataset directory `--dataset_root_dir /path/to/your/dataset`. There are prebuilt lists under [./patchnetvlad/dataset_imagenames](./patchnetvlad/dataset_imagenames).

### Generate image list from your own dataset
Navigate to tool folder
```bash
cd patchnetvlad/tools
```
Run the list generator
```bash
python gen_image_file_list.py --dataset_root_dir /path/to/your/dataset/root \
                       --out_file ../dataset_imagenames/your_dataset_index.txt
```
> Replace `_index` with `_query` if you want to run the benchmark and matching stream.

## Feature Extraction and Matching
### Extract feature from database
```bash
python feature_extract.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_file_path your_dataset_index.txt \
  --dataset_root_dir /path/to/your/dataset/root \
  --output_features_dir patchnetvlad/output_features/your_dataset_index
```
> It will download pre-trained models in the first run.

### Extract feature from query
```bash
python feature_extract.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_file_path your_dataset_query.txt \
  --dataset_root_dir /path/to/your/dataset/root \
  --output_features_dir patchnetvlad/output_features/your_dataset_query
```
> You can combine above two scripts at once since `performance.ini` takes couple hours.


### Feature Matching and Benchmark
To run the benchmark, make sure to set the `ground_true_path`
```bash
python feature_match.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_root_dir /path/to/your/dataset/root \
  --query_file_path your_dataset_query.txt \
  --index_file_path your_dataset_index.txt \
  --query_input_features_dir patchnetvlad/output_features/your_dataset_query \
  --index_input_features_dir patchnetvlad/output_features/your_dataset_index \
  --result_save_folder patchnetvlad/results/your_dataset \
  --ground_truth_path patchnetvlad/dataset_gt_files/your_dataset_test.npz
```
> To create your own ground true file, please check the original [README.md](./README.md).

### Feature matching (2 images)
```bash
python match_two.py \
--config_path patchnetvlad/configs/performance.ini \
--first_im_path patchnetvlad/example_images/tokyo_query.jpg \
--second_im_path patchnetvlad/example_images/tokyo_db.png
```

### Real-time webcam feature matching
Take the first frame from webcam and match local features in stream.

Press `n` to setup new reference image. (Known BUG that causes crash.)

Press `q` to quit.
```bash
python match_realtime.py
```

# Feature Matching from Webcam Stream
### Extract feature from database

```bash
python feature_extract.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_file_path example_imageNames_index.txt \
  --dataset_root_dir /path/to/your/dataset/root \
  --output_features_dir patchnetvlad/output_features/your_dataset_index
```

> Make sure you run **above script** to build feature files first.
```bash
python feature_match_realtime.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_root_dir /path/to/your/dataset/root \
  --index_file_path example_imageNames_index.txt \
  --index_input_features_dir patchnetvlad/output_features/your_dataset_index
```

# Feature Matching from dataset images
### Extract features from database
```bash
python feature_extract.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_file_path your_dataset_index.txt \
  --dataset_root_dir /path/to/your/dataset/root \
  --output_features_dir patchnetvlad/output_features/your_dataset_index
```

### Match feature from query dataset, frame-by-frame/stream
```bash
python feature_match_read.py \
  --config_path patchnetvlad/configs/performance.ini \
  --index_file_path your_dataset_index.txt \
  --index_input_features_dir patchnetvlad/output_features/your_dataset_index \
  --dataset_root_dir /path/to/your/dataset/root \
  --query_file_path /relative/path/to/your/query/image
  
```

# FAQ
1. `asert d    self.d`, Check the configuration file. Make sure the feature files are built from the same configuration or it will throw this error.