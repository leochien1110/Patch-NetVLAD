
# Table of Contents
1. [Environment Setup](#env-setup)
2. [Basic Operation](#basic-operation)
3. [Feature Matching Webcam Stream](#feature-matching-webcam-stream)
4. [FAQ](#faq)

# Environment Setup
Install Aconda or Miniconda from the offical website.
### Build Conda Environment
> Change the `cudatoolit=________` for your CUDA version.

`conda create -n patchnetvlad python=3.8 numpy pytorch-gpu torchvision \
cudatoolkit=10.2 natsort tqdm opencv pillow scikit-learn faiss matplotlib-base \
-c conda-forge`

### Activate conda environment

`conda activate patchnetvlad`

# Basic Operation
The configuration must be the same for feature extraction and feature matching.
1. Performance: the most robust recognition but slow for real-time process
2. performance: Fast for both setting up database and real-time process. Less accurate.
3. performance: Use less space in the drive and fastest. The most inaccurate.

> For performance configuration, make sure to have enough disk space or the code might stop without enough space. For Pittsburgh30k, it takes about 500GB.

Setup your own dataset directory `--dataset_root_dir=/path/to/your/dataset`

### Generate image list from your own dataset
Navigate to tool folder
```bash
cd patchnetvlad/tools
```
Run the list generator
```bash
python gen_image_file_list.py --dataset_root_dir=/path/to/your/dataset/root \
                       --out_file ../dataset_imagenames/your_dataset_index.txt
```
> Replace `_index` with `_query` if you want to run the benchmark and matching stream.

### Extract feature from database
```bash
python feature_extract.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_file_path=your_dataset_index.txt \
  --dataset_root_dir=/path/to/your/dataset/root \
  --output_features_dir patchnetvlad/output_features/your_dataset_index
```

### Extract feature from query
```bash
python feature_extract.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_file_path=your_dataset_query.txt \
  --dataset_root_dir=/path/to/your/dataset/root \
  --output_features_dir patchnetvlad/output_features/your_dataset_query
```

### Or combine above two scripts at once since performance.ini takes couple hours
```bash
python feature_extract.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_file_path=your_dataset_index.txt \
  --dataset_root_dir=/path/to/your/dataset/root \
  --output_features_dir patchnetvlad/output_features/your_dataset_index \
&& python feature_extract.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_file_path=your_dataset_query.txt \
  --dataset_root_dir=/path/to/your/dataset/root \
  --output_features_dir patchnetvlad/output_features/your_dataset_query
```

### Feature Matching and Benchmark
To run the benchmark, make sure to set the `ground_true_path`
```bash
python feature_match.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_root_dir=/path/to/your/dataset/root \
  --query_file_path=your_dataset_query.txt \
  --index_file_path=your_dataset_index.txt \
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
--first_im_path=patchnetvlad/example_images/tokyo_query.jpg \
--second_im_path=patchnetvlad/example_images/tokyo_db.png
```

### Real-time webcam feature matching
Take the first frame from webcam and match local features in stream.

Press `n` to setup new reference image. (Known BUG that causes crash.)

Press `q` to quit.
```bash
python match_realtime.py
```

# Feature Matching Webcam Stream
### Extract feature from database

```bash
python feature_extract.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_file_path=example_imageNames_index.txt \
  --dataset_root_dir=/path/to/your/dataset/root \
  --output_features_dir patchnetvlad/output_features/your_dataset_index
```

Make sure to run **above script** to build feature files first.
```bash
python feature_match_realtime.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_root_dir=/path/to/your/dataset/root \
  --index_file_path=example_imageNames_index.txt \
  --index_input_features_dir patchnetvlad/output_features/your_dataset_index
```

## Read from dataset
### Extract features from database
```bash
python feature_extract.py \
  --config_path patchnetvlad/configs/performance.ini \
  --dataset_file_path=your_dataset_index.txt \
  --dataset_root_dir=/path/to/your/dataset/root \
  --output_features_dir patchnetvlad/output_features/your_dataset_index
```

### Match feature from query dataset, frame-by-frame/stream
```bash
python feature_match_read.py \
  --config_path patchnetvlad/configs/performance.ini \
  --index_file_path=your_dataset_index.txt \
  --index_input_features_dir patchnetvlad/output_features/your_dataset_index \
  --query_file_path=/path/to/your/query/image \
  --dataset_root_dir=/path/to/your/dataset/root
```

# FAQ
1. `asert d == self.d`, Check the configuration file. Make sure the feature files are built from the same configuration or it will throw this error.