#!/usr/bin/env python

'''
MIT License

Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Performs place recognition using a two-stage image retrieval pipeline, where
the first step collects the top 100 database candidates and then geometric
verification produces the top 1 best match for every query. In this code, query
images are the image from webcam. Change the video index to select your webcam.

Requires feature_extract.py to be run first, on a folder of index/database
images.

Code already supports the datasets of Nordland, Pittsburgh 30k and Tokyo247,
please run tools/genImageListFile to create new imageNames files with your
filepaths pointing to where you saved these datasets (or, edit the text files
to remove the prefix and insert your own prefix).
'''


from __future__ import print_function

import os
import time
import argparse
import configparser
from os.path import join, isfile
from os.path import exists
from os import makedirs
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import faiss
from tqdm.auto import tqdm
import cv2
from PIL import Image

from patchnetvlad.tools.datasets import PlaceDataset, input_transform
from patchnetvlad.tools.patch_matcher import PatchMatcher
from patchnetvlad.models.local_matcher import normalise_func, calc_keypoint_centers_from_patches as calc_keypoint_centers_from_patches
from patchnetvlad.models.models_generic import get_backend, get_model, get_pca_encoding
from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR


def apply_patch_weights(input_scores, num_patches, patch_weights):
    output_score = 0
    if len(patch_weights) != num_patches:
        raise ValueError('The number of patch weights must equal the number of patches used')
    for i in range(num_patches):
        output_score = output_score + (patch_weights[i] * input_scores[i])
    return output_score


def plot_two(im1, im2, inlier_keypoints_one, inlier_keypoints_two, score, image_index, window_name):
    
    # Draw keypoints
    kp_all1 = []
    kp_all2 = []
    matches_all = []
    for this_inlier_keypoints_one, this_inlier_keypoints_two in zip(inlier_keypoints_one, inlier_keypoints_two):
        for i in range(this_inlier_keypoints_one.shape[0]):
            kp_all1.append(cv2.KeyPoint(this_inlier_keypoints_one[i, 0].astype(float), this_inlier_keypoints_one[i, 1].astype(float), 1, -1, 0, 0, -1))
            kp_all2.append(cv2.KeyPoint(this_inlier_keypoints_two[i, 0].astype(float), this_inlier_keypoints_two[i, 1].astype(float), 1, -1, 0, 0, -1))
            matches_all.append(cv2.DMatch(i, i, 0))

    im_allpatch_matches = cv2.drawMatches(im1, kp_all1, im2, kp_all2,
                                          matches_all, None, matchColor=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.putText(im_allpatch_matches, f"Retrieved Image: {image_index} ({score:.5})", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 2)
    cv2.imshow(window_name, im_allpatch_matches)


def feature_extract(model, device, config, img):

    pool_size = int(config['global_params']['num_pcs'])

    model.eval()

    it = input_transform((int(config['feature_extract']['imageresizeH']), int(config['feature_extract']['imageresizeW'])))

    im_one_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    im_one_pil = it(im_one_pil).unsqueeze(0)

    input_data = im_one_pil.to(device)

    tqdm.write('====> Extracting Features')
    with torch.no_grad():
        image_encoding = model.encoder(input_data)

        vlad_local, vlad_global = model.pool(image_encoding)
        vlad_global_pca = get_pca_encoding(model, vlad_global).cpu().numpy()

        local_feats_one = []
        for this_iter, this_local in enumerate(vlad_local):
            this_local_feats = get_pca_encoding(model, this_local.permute(2, 0, 1).reshape(-1, this_local.size(1))). \
                reshape(this_local.size(2), this_local.size(0), pool_size).permute(1, 2, 0)
            local_feats_one.append(torch.transpose(this_local_feats[0, :, :], 0, 1))

    return local_feats_one, vlad_global_pca


def feature_match(eval_set, device, opt, config, im_query, local_feat, query_global_feat):
    # input_query_local_features_prefix = join(opt.query_input_features_dir, 'patchfeats')
    # input_query_global_features_prefix = join(opt.query_input_features_dir, 'globalfeats.npy')
    input_index_local_features_prefix = join(opt.index_input_features_dir, 'patchfeats')
    input_index_global_features_prefix = join(opt.index_input_features_dir, 'globalfeats.npy')
    
    pool_size = query_global_feat.shape[1]
    dbFeat = np.load(input_index_global_features_prefix)

    if dbFeat.dtype != np.float32:
        query_global_feat = query_global_feat.astype('float32')
        dbFeat = dbFeat.astype('float32')

    tqdm.write('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    # noinspection PyArgumentList
    faiss_index.add(dbFeat)

    n_values = []
    for n_value in config['feature_match']['n_values_all'].split(","):  # remove all instances of n that are bigger than maxK
        n_values.append(int(n_value))

    tqdm.write('====> Matching Global Features')
    if config['feature_match']['pred_input_path'] != 'None':
        predictions = np.load(config['feature_match']['pred_input_path'])  # optionally load predictions from a np file
    else:
        # noinspection PyArgumentList
        # _, predictions = faiss_index.search(global_feat, min(len(global_feat), max(n_values)))
        _, predictions = faiss_index.search(query_global_feat, 3)

    tqdm.write('====> Loading patch param from config')
    patch_sizes = [int(s) for s in config['global_params']['patch_sizes'].split(",")]
    strides = [int(s) for s in config['global_params']['strides'].split(",")]
    patch_weights = np.array(config['feature_match']['patchWeights2Use'].split(",")).astype(float)

    all_keypoints = []
    all_indices = []

    tqdm.write('====> Matching Local Features')
    for patch_size, stride in zip(patch_sizes, strides):
        # we currently only provide support for square patches, but this can be easily modified for future works
        keypoints, indices = calc_keypoint_centers_from_patches(config['feature_match'], patch_size, patch_size, stride, stride)
        all_keypoints.append(keypoints)
        all_indices.append(indices)

    matcher = PatchMatcher(config['feature_match']['matcher'], patch_sizes, strides, all_keypoints,
                           all_indices)
    reordered_preds = []
    for q_idx, pred in enumerate(tqdm(predictions, leave=False, desc='Patch compare pred')):
        diffs = np.zeros((predictions.shape[1], len(patch_sizes)))
        # we pre-transpose here to save compute speed
        for k, candidate in enumerate(pred):
            image_name_index = os.path.splitext(os.path.basename(eval_set.images[candidate]))[0]
            dbfeat = []
            for patch_size in patch_sizes:
                dbfilename = input_index_local_features_prefix + '_' + 'psize{}_'.format(patch_size) + image_name_index + '.npy'
                dbfeat.append(torch.tensor(np.load(dbfilename), device=device))

            if k == 0:
                # Get the NetVLAD top candidate's keypoints and score
                scores, keypoints_net_one, keypoints_net_two = matcher.match(local_feat, dbfeat)
                diffs[k, :] = scores
                score_net = -apply_patch_weights(scores, len(patch_sizes), patch_weights)
                print(f"NetVLAD: Similarity score between the two images is: {score_net:.5f}. Larger is better.")
            else:
                diffs[k, :], _, _ = matcher.match(local_feat, dbfeat)
            
        diffs = normalise_func(diffs, len(patch_sizes), patch_weights)
        cand_sorted = np.argsort(diffs)
        reordered_preds.append(pred[cand_sorted])

    # Top candidates from two methods
    image_name_index_net = os.path.splitext(os.path.basename(eval_set.images[predictions[0][0]]))[0]
    image_name_index_patch = os.path.splitext(os.path.basename(eval_set.images[reordered_preds[0][0]]))[0]

    # Get the Patch-NetVLAD top candidate's keypoints and score
    dbfeat = []
    for patch_size in patch_sizes:
        dbfilename = input_index_local_features_prefix + '_' + 'psize{}_'.format(patch_size) + image_name_index_patch + '.npy'
        dbfeat.append(torch.tensor(np.load(dbfilename), device=device))
    scores, keypoints_patch_one, keypoints_patch_two = matcher.match(local_feat, dbfeat)
    score_patch = -apply_patch_weights(scores, len(patch_sizes), patch_weights)
    print(f"Patch-NetVLAD: Similarity score between the two images is: {score_patch:.5f}. Larger is better.")
    
    print('predictions: ', predictions[0])
    print('reordered_preds: ', reordered_preds[0])

    # Show the most possible retrieved image
    image_list_array = np.array(eval_set.images)        
    im_db_net = cv2.imread(image_list_array[predictions[0]][0])
    im_db_patch = cv2.imread(image_list_array[reordered_preds[0]][0])

    # using cv2 for their in-built keypoint correspondence plotting tools
    im_query = cv2.resize(im_query, (int(config['feature_extract']['imageresizeW']), int(config['feature_extract']['imageresizeH'])))
    im_db_net = cv2.resize(im_db_net, (int(config['feature_extract']['imageresizeW']), int(config['feature_extract']['imageresizeH'])))
    im_db_patch = cv2.resize(im_db_patch, (int(config['feature_extract']['imageresizeW']), int(config['feature_extract']['imageresizeH'])))
    # cv2 resize slightly different from torch, but for visualisation only not a big problem

    if config['feature_match']['matcher'] == 'RANSAC':
        # Draw local matches
        plot_two(im_query, im_db_net, keypoints_net_one, keypoints_net_two, score_net, image_name_index_net, 'NetVLAD')
        plot_two(im_query, im_db_patch, keypoints_patch_one, keypoints_patch_two, score_patch, image_name_index_patch, 'Patch-NetVLAD')
    else:
        cv2.imshow('NetVLAD Top Match', im_db_net)
        cv2.imshow('Patch-NetVLAD Top Match', im_db_patch)
    
    
def main():
    parser = argparse.ArgumentParser(description='Patch-NetVLAD-Feature-Match')
    parser.add_argument('--config_path', type=str, default=join(PATCHNETVLAD_ROOT_DIR, 'configs/performance.ini'),
                        help='File name (with extension) to an ini file that stores most of the configuration data for patch-netvlad')
    parser.add_argument('--dataset_root_dir', type=str, default='',
                        help='If the files in query_file_path and index_file_path are relative, use dataset_root_dir as prefix.')
    parser.add_argument('--index_file_path', type=str, required=True,
                        help='Path (with extension) to a text file that stores the save location and name of all database images in the dataset')
    parser.add_argument('--index_input_features_dir', type=str, required=True,
                        help='Path to load all database patch-netvlad features')
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')

    opt = parser.parse_args()
    print(opt)

    # load config file
    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)
    
    # check GPU/cuda
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    # load model
    encoder_dim, encoder = get_backend()

    # must load from a resume to do extraction
    resume_ckpt = config['global_params']['resumePath'] + config['global_params']['num_pcs'] + '.pth.tar'

    # backup: try whether resume_ckpt is relative to script path
    if not isfile(resume_ckpt):
        resume_ckpt = join(PATCHNETVLAD_ROOT_DIR, resume_ckpt)
        if not isfile(resume_ckpt):
            from download_models import download_all_models
            download_all_models(ask_for_permission=True)

    if isfile(resume_ckpt):
        print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        assert checkpoint['state_dict']['WPCA.0.bias'].shape[0] == int(config['global_params']['num_pcs'])
        config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])

        model = get_model(encoder, encoder_dim, opt, config['global_params'], append_pca_layer=True)

        if int(config['global_params']['nGPU']) > 1 and torch.cuda.device_count() > 1:
            model.encoder = nn.DataParallel(model.encoder)
            model.pool = nn.DataParallel(model.pool)

        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        print("=> loaded checkpoint '{}'".format(resume_ckpt, ))
    else:
        raise FileNotFoundError("=> no checkpoint found at '{}'".format(resume_ckpt))

    # check database path
    if not os.path.isfile(opt.index_file_path):
        opt.index_file_path = join(PATCHNETVLAD_ROOT_DIR, 'dataset_imagenames', opt.index_file_path)

    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start = time.time()

    while(True):
        _, frame = vid.read()

        # extract query feature
        local_feat, global_feat = feature_extract(model, device, config, frame)

        dataset = PlaceDataset(None, opt.index_file_path, opt.dataset_root_dir, None, config['feature_extract'])
        # match feature
        feature_match(dataset, device, opt, config, frame, local_feat, global_feat)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('n'):
            print('why press \'n\'?')

        end = time.time()
        elapse = end - start
        start = end
        print(f"FPS: {1/elapse}")
        cv2.putText(frame, f"FPS: {1/elapse}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,128,128), 1)
        cv2.imshow('Query Image', frame)
        
    vid.release()
    cv2.destroyAllWindows()
    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
                              # memory after runs

if __name__ == "__main__":
    main()