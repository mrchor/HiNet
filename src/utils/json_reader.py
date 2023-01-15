#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     json_reader
   Author :       mrchor
-------------------------------------------------
"""
import json
import tensorflow as tf


def load_json(json_file_path):
    with open(json_file_path, "r") as config_file:
        try:
            json_conf = json.load(config_file)
            return json_conf
        except Exception:
            pass
            tf.logging.error("load json file %s error" % json_file_path)

if __name__ == '__main__':
    pass