#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: cofacto_train.py
Author: leowan
Date: 2018/06/17 07:47:45
"""

import os
import glob
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from joblib import Parallel, delayed

import cofacto_util as util
import cofacto
import rec_eval

DATA_DIR = '../data/'
N_COMPONENTS = 100
DATA_SET_NAME = 'UIDSID'
FN_UID_SIDS = 'uid_sids.txt'

N_COMPONENTS = 100
N_JOBS = 30
K_NS = 1
scale = 0.03
max_iter = 20
lam_theta = lam_beta = 1e-5 * scale
lam_gamma = 1e-5
c0 = 1. * scale
c1 = 10. * scale

save_dir = os.path.join(DATA_DIR, '%s_ns%d_scale%1.2E' % (DATA_SET_NAME, K_NS, scale))

idx2uid, uid2idx, idx2sid, sid2idx = util.make_index(os.path.join(DATA_DIR, FN_UID_SIDS))
uid_sids_dict = util.make_uid_sids_dict(os.path.join(DATA_DIR, FN_UID_SIDS))

def train(sppmi_mat, data_dir=DATA_DIR):
    '''
        Train
    '''
    # load file
    n_users = len(uid2idx)
    print('n_users:{}'.format(n_users))
    n_items = len(sid2idx)
    print('n_items:{}'.format(n_items))
    fn_train = os.path.join(data_dir, 'data_train.txt')
    fn_dev = os.path.join(data_dir, 'data_dev.txt')
    fn_test = os.path.join(data_dir, 'data_test.txt')
    train_data = util.load_input_data(fn_train, shape=(n_users, n_items))
    vad_data = util.load_input_data(fn_dev, shape=(n_users, n_items))
    test_data = util.load_input_data(fn_test, shape=(n_users, n_items))
    # train model
    coder = cofacto.CoFacto(n_components=N_COMPONENTS, max_iter=max_iter, batch_size=1000, init_std=0.01, n_jobs=N_JOBS, 
        random_state=98765, save_params=True, save_dir=save_dir, early_stopping=True, verbose=True, 
        lam_theta=lam_theta, lam_beta=lam_beta, lam_gamma=lam_gamma, c0=c0, c1=c1)
    coder.fit(train_data, sppmi_mat, vad_data=vad_data, batch_users=5000, k=100)
    # test model
    n_params = len(glob.glob(os.path.join(save_dir, '*.npz')))
    last_iter_num = n_params - 1
    params = np.load(os.path.join(save_dir, 'CoFacto_K%d_iter%d.npz' % (N_COMPONENTS, last_iter_num)))
    U, V = params['U'], params['V']
    print('Test Recall@20: %.4f' % rec_eval.recall_at_k(train_data, test_data, U, V, k=20, vad_data=vad_data))
    print('Test Recall@50: %.4f' % rec_eval.recall_at_k(train_data, test_data, U, V, k=50, vad_data=vad_data))
    print('Test NDCG@100: %.4f' % rec_eval.normalized_dcg_at_k(train_data, test_data, U, V, k=100, vad_data=vad_data))
    print('Test MAP@100: %.4f' % rec_eval.map_at_k(train_data, test_data, U, V, k=100, vad_data=vad_data))
    # save
    model_save_fn = os.path.join(DATA_DIR, 'Model_K{}_{}.npz'.format(N_COMPONENTS, DATA_SET_NAME))
    np.savez(model_save_fn, U=U, V=V)
    print('saved')