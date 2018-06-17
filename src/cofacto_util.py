#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: cofacto_util.py
Author: leowan
Date: 2018/06/10 20:17:56
"""

import os
import numpy as np
from scipy import sparse


def load_input_data(fn_data, shape=(-1, -1)):
    """
        Model input data loader
        Params:
            fn_data : data file name
            shape : (n_users, n_items)
    """
    import csv
    rows = []
    cols = []
    with open(fn_data, 'r') as fd:
        elems = csv.reader(fd, delimiter='\t')
        for uididx, sididx in elems:
            rows.append(int(uididx))
            cols.append(int(sididx))
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data


def yield_uid_sids_from_file(fn):
    """
        Generator
    """
    fd = open(fn, "r")
    for line in fd:
        elems = line.rstrip().split(',')
        uid = elems[0]
        sids = set()
        for sid in elems[1].split('|'):
            if sid != '':
                sids.add(sid)
        yield uid, sids
    fd.close()


def make_sid_count_dict(fn):
    """
        Count
    """
    from collections import defaultdict
    sid_cnt_dict = defaultdict(int)
    for uid, sids in yield_uid_sids_from_file(fn):
        for sid in sids:
            sid_cnt_dict[sid] += 1
    return sid_cnt_dict


def make_sid_index(sid_cnt_dict):
    """
        Sort
    """
    import operator
    return sorted(sid_cnt_dict.items(), key=operator.itemgetter(1), reverse=True)


def make_index(fn):
    """
        Index
    """
    idx2uid = {}
    uid2idx = {}
    idx2sid = {}
    sid2idx = {}
    idx = 0
    for uid, _ in yield_uid_sids_from_file(fn):
        idx2uid[idx] = uid
        uid2idx[uid] = idx
        idx += 1
    sid_index_arr = make_sid_index(make_sid_count_dict(fn))
    idx = 0
    for sid, _ in sid_index_arr:
        idx2sid[idx] = sid
        sid2idx[sid] = idx
        idx += 1
    return tuple([idx2uid, uid2idx, idx2sid, sid2idx])


def make_uid_sids_dict(fn):
    """
        uid -> sids
    """
    uid_sids_dict = {}
    for uid, sids in yield_uid_sids_from_file(fn):
        uid_sids_dict[uid] = sids
    return uid_sids_dict


def make_train_dev_test_files(idx2uid, uid2idx, idx2sid, sid2idx, uid_sids_dict, data_dir='./'):
    """
        Create train dev files from uid sids info
    """
    from sklearn.model_selection import train_test_split
    import numpy as np
    fn_train = os.path.join(data_dir, 'data_train.txt')
    fn_dev = os.path.join(data_dir, 'data_dev.txt')
    fn_test = os.path.join(data_dir, 'data_test.txt')
    fdt = open(fn_train, 'w')
    fdv = open(fn_dev, 'w')
    fdd = open(fn_test, 'w')
    for uid, sids in uid_sids_dict.items():
        pos_sids = list(sids)
        pos_ratings = np.ones(len(pos_sids), dtype=int)
        X = pos_sids
        y = pos_ratings.tolist()
        X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=0.15)
        X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.2)
        for xt, yt in zip(X_train, y_train):
            fdt.write('{}\t{}\n'.format(uid2idx[uid], sid2idx[xt]))
        for xt, yt in zip(X_train, y_train):
            fdv.write('{}\t{}\n'.format(uid2idx[uid], sid2idx[xt]))
        for xd, yd in zip(X_test, y_test):
            fdd.write('{}\t{}\n'.format(uid2idx[uid], sid2idx[xd]))
    fdt.close()
    fdv.close()
    fdd.close()
    return fn_train, fn_dev, fn_test


def count_user_item_entry(fn):
    """
        Count user item entry in file
    """
    import csv
    entry_cnt = 0
    max_user_idx = 0
    max_item_idx = 0
    with open(fn, 'r') as f:
        elems = csv.reader(f, delimiter='\t')
        for uididx, sididx in elems:
            entry_cnt += 1
            max_user_idx = max(max_user_idx, int(uididx))
            max_item_idx = max(max_item_idx, int(sididx))
    return max_user_idx, max_item_idx, entry_cnt