#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: cofacto_preprocess.py
Author: leowan
Date: 2018/06/17 11:04:48
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import itertools

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from joblib import Parallel, delayed

import cofacto_util as util
import cofacto
import rec_eval

DATA_DIR = '../data/'
DATA_SET_NAME = 'UIDSID'
FN_UID_SIDS = 'sample_uid_sids.txt'
IS_ILLUSTRATED = False
N_JOBS = 30
K_NS = 1
COMAT_DATA_FN = 'comat_data.npy'
COMAT_INDEX_FN = 'comat_indices.npy'
COMAT_PTR_FN = 'comat_indptr.npy'
_COOD_MID_FILE_FORMAT = 'coo_%d_%d.npy'

if IS_ILLUSTRATED:
    import pandas as pd
    import seaborn as sns
    sns.set(context="paper", font_scale=1.5, rc={"lines.linewidth": 2}, font='DejaVu Serif')


idx2uid, uid2idx, idx2sid, sid2idx = util.make_index(os.path.join(DATA_DIR, FN_UID_SIDS))
uid_sids_dict = util.make_uid_sids_dict(os.path.join(DATA_DIR, FN_UID_SIDS))


def _coord_batch(lo, hi, train_data):
    '''
        Compute item item coordinates in batch
    '''
    rows = []
    cols = []
    print('Genereting coord batched from %d to %d'% (lo, hi))
    for u in range(lo, hi):
        for w, c in itertools.permutations(train_data[u].nonzero()[1], 2):
            rows.append(w)
            cols.append(c)
    np.save(os.path.join(DATA_DIR, _COOD_MID_FILE_FORMAT % (lo, hi)),
            np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1))


def load_co_matrix():
    '''
        Load Co-occurrence Matrix
    '''
    n_users = len(uid2idx)
    print('n_users:{}'.format(n_users))
    n_items = len(sid2idx)
    print('n_items:{}'.format(n_items))
    data = np.load(os.path.join(DATA_DIR, COMAT_DATA_FN))
    indices = np.load(os.path.join(DATA_DIR, COMAT_INDEX_FN))
    indptr = np.load(os.path.join(DATA_DIR, COMAT_PTR_FN))
    mat = sparse.csr_matrix((data, indices, indptr), shape=(n_items, n_items))
    print('co-occurrence matrix sparcity:{}'.format(float(mat.nnz) / np.prod(mat.shape)))
    return mat


def make_sppmi_matrix(comat, k_ns=1):
    '''
        Compute Shifted Positive Pairwise Mutual Information Matrix (log)
        Params:
            k_ns: number of negative samples
    '''
    def get_row(M, i):
        '''
            Get row data from sparse matrix
        '''
        lo, hi = M.indptr[i], M.indptr[i + 1]
        return lo, hi, M.data[lo:hi], M.indices[lo:hi]

    n_items = len(sid2idx)
    X = comat
    count = np.asarray(X.sum(axis=1)).ravel()
    n_pairs = X.data.sum()
    ### math log
    M = X.copy()
    for i in range(n_items):
        lo, hi, d, idx = get_row(M, i)
        M.data[lo:hi] = np.log(d * n_pairs / (count[i] * count[idx]))
    M.data[M.data < 0] = 0
    M.eliminate_zeros()
    print('PPMI matrix M sparcity:{}'.format(float(M.nnz) / np.prod(M.shape)))
    ### shift
    M_ns = M.copy()
    if k_ns > 1:
        offset = np.log(k_ns)
    else:
        offset = 0.

    M_ns.data -= offset
    M_ns.data[M_ns.data < 0] = 0
    M_ns.eliminate_zeros()
    # illustrate
    fig = plt.figure()
    plt.hist(M_ns.data, bins=50)
    plt.yscale('log')
    if IS_ILLUSTRATED:
        fig.show()
    else:
        fig.savefig('co_ocur_hist.png')
    print('SPPMI matrix M_ns sparcity:{}'.format(float(M_ns.nnz) / np.prod(M_ns.shape)))
    return M_ns


def preprocess():
    '''
        Proprocess
    '''
    fn_train, fn_dev, fn_test = util.make_train_dev_test_files(idx2uid, uid2idx, idx2sid, sid2idx, uid_sids_dict, data_dir=DATA_DIR)
    print(fn_train, fn_dev, fn_test)
    
    n_users = len(uid2idx)
    print('n_users:{}'.format(n_users))
    n_items = len(sid2idx)
    print('n_items:{}'.format(n_items))

    ### Construct the positive pairwise mutual information (PPMI) matrix
    train_data = util.load_input_data(fn_train, shape=(n_users, n_items))
    watches_per_item = np.asarray(train_data.astype('int64').sum(axis=0)).ravel()
    print("The mean (median) watches per item is %d (%d)" % (watches_per_item.mean(), np.median(watches_per_item)))
    user_activity = np.asarray(train_data.sum(axis=1)).ravel()
    print("The mean (median) items each user wathced is %d (%d)" % (user_activity.mean(), np.median(user_activity)))
    # illustrate user->item count stat
    fig = plt.figure()
    plt.semilogx(1 + np.arange(n_users), -np.sort(-user_activity), 'o')
    plt.ylabel('Number of items that this user clicked on')
    plt.xlabel('User rank by number of consumed items')
    if IS_ILLUSTRATED:
        plt.show()
    else:
        fig.savefig('user_rank.png')
    # illustrate item->user count stat
    fig = plt.figure()
    plt.semilogx(1 + np.arange(n_items), -np.sort(-watches_per_item), 'o')
    plt.ylabel('Number of users who watched this item')
    plt.xlabel('Item rank by number of watches')
    if IS_ILLUSTRATED:
        plt.show()
    else:
        fig.savefig('item_rank.png')
    ## Generate co-occurrence matrix based on the user's entire watching history
    batch_size = 5000
    start_idx = list(range(0, n_users, batch_size))
    end_idx = start_idx[1:] + [n_users]
    Parallel(n_jobs=N_JOBS)(delayed(_coord_batch)(lo, hi, train_data) for lo, hi in zip(start_idx, end_idx))
    mat = sparse.csr_matrix((n_items, n_items), dtype='float32') # co-occurrence matrix
    for lo, hi in zip(start_idx, end_idx):
        coords = np.load(os.path.join(DATA_DIR, _COOD_MID_FILE_FORMAT % (lo, hi)))
        rows = coords[:, 0]
        cols = coords[:, 1]
        tmp = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(n_items, n_items), dtype='float32').tocsr()
        mat = mat + tmp
        print("Users %d to %d finished" % (lo, hi))
        sys.stdout.flush()
    ## Persist
    np.save(os.path.join(DATA_DIR, COMAT_DATA_FN), mat.data)
    np.save(os.path.join(DATA_DIR, COMAT_INDEX_FN), mat.indices)
    np.save(os.path.join(DATA_DIR, COMAT_PTR_FN), mat.indptr)
    print('co-occurrence matrix sparcity:{}'.format(float(mat.nnz) / np.prod(mat.shape)))
    return mat