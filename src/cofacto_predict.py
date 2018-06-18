#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: cofacto_predict.py
Author: leowan
Date: 2018/06/17 07:50:50
"""

import os
import sys
import numpy as np

import cofacto_util as util

DATA_DIR = '../data/'
N_COMPONENTS = 100
DATA_SET_NAME = 'UIDSID'
FN_UID_SIDS = 'uid_sids.txt'

idx2uid, uid2idx, idx2sid, sid2idx = util.make_index(os.path.join(DATA_DIR, FN_UID_SIDS))
uid_sids_dict = util.make_uid_sids_dict(os.path.join(DATA_DIR, FN_UID_SIDS))


def load_model(data_dir=DATA_DIR, n_components=N_COMPONENTS, data_set_name=DATA_SET_NAME):
    '''
        Load
    '''
    model_save_fn = os.path.join(data_dir, 'Model_K{}_{}.npz'.format(n_components, data_set_name))
    npz = np.load(model_save_fn)
    return npz['U'], npz['V']


def uid_history_scores(uid, U, V):
    '''
        Dump scores for user history
    '''
    scores = []
    sids = uid_sids_dict[uid]
    for sid in sids:
        scores.append(np.dot(U[uid2idx[uid]], V[sid2idx[sid]]))
    return scores, list(sids)


def topk_for_uid(uid, U, V, topk=50, sid_prefix=None):
    '''
        Topk
    '''
    uv_scores = np.matmul(U[uid2idx[uid]], V[:, :].T)
    topk_sids = []
    topk_scores = []
    if sid_prefix is None:
        topk_idx = np.argpartition(-uv_scores, topk)[:topk]
        topk_scores = uv_scores[topk_idx]
        for idx in topk_idx.tolist():
            topk_sids.append(idx2sid[idx])
    else:
        thresh = 5000
        topk_cand_idx = np.argpartition(-uv_scores, thresh)[:thresh]
        topk_cand_scores = uv_scores[topk_cand_idx]
        topk_cand_score_idx = list(zip(topk_cand_scores, topk_cand_idx))
        topk_cand_score_idx = sorted(topk_cand_score_idx, key=lambda x:x[0], reverse=True)
        for score, idx in topk_cand_score_idx:
            sid = idx2sid[idx]
            if sid[:len(sid_prefix)] == sid_prefix:
                topk_sids.append(sid)
                topk_scores.append(score)
                if len(topk_sids) >= topk:
                    break
    return topk_sids, topk_scores


def difference_and_intersection(uid, topk_sids):
    '''
        Difference and Intersection
    '''
    recommend_sids = list(set(topk_sids).difference(uid_sids_dict[uid]))
    intersected_sids = list(set(topk_sids).intersection(uid_sids_dict[uid]))
    return recommend_sids, intersected_sids