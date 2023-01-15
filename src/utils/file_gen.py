#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     file_gen
   Author :       mrchor
-------------------------------------------------
"""
import glob

def get_local_path(base_path):
    files = []
    for file_path in base_path.strip().split(','):
        files += glob.glob('%s/*' % file_path)
    return files

if __name__ == '__main__':
    pass