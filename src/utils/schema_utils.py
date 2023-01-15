# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     feature_utils
   Author :       mrchor
-------------------------------------------------
"""
import tensorflow as tf

fc = tf.feature_column

def _generate_categorical_column_with_hash_bucket(features, dimension, hash_bucket_size, dtype='int64'):
    bucket = [
        fc.categorical_column_with_hash_bucket(key, hash_bucket_size, eval('tf.{}'.format(dtype)))
        for key in features
    ]
    fcs = fc.shared_embedding_columns(categorical_columns=bucket, dimension=dimension)
    return {'columns': fcs}

def _generate_raw_numeric_columns(features):
    fcs = [
        fc.numeric_column(key)
        for key in features
    ]
    return {'columns': fcs}

def _generate_embedding_columns(features, dimension, default_value):
    categorical_columns = [
        fc.categorical_column_with_vocabulary_list(key, features[key], default_value=default_value)
        for key in features
    ]
    fcs = [
        fc.embedding_column(column, dimension=dimension)
        for column in categorical_columns
    ]
    res = dict()
    res["dimension"] = dimension
    res["columns"] = fcs
    return res

def get_feature_transform(js):
    if not isinstance(js, dict):
        assert TypeError("js is not dict!")
    feature_schema = {}
    for group_name,v in js['data_schema']['feature_group'].items():
        dtype = v['dtype']
        operator = v['operator']
        reshape = v['reshape']
        dimension = v['dimension']
        features = v['features']
        default_value = v['default_value']
        if operator == 'hash_bucket':
            feature_schema[group_name] = _generate_categorical_column_with_hash_bucket(features, dimension, v['hash_bucket_size'], dtype=dtype)
            feature_schema[group_name]["reshape"] = reshape
            feature_schema[group_name]["embedding_size"] = dimension
            if 'max_len' in v:
                feature_schema[group_name]["max_len"] = v['max_len']
        elif operator == 'embedding':
            feature_schema[group_name] = _generate_embedding_columns(features, dimension, default_value)
            feature_schema[group_name]["reshape"] = reshape
            feature_schema[group_name]["embedding_size"] = dimension
        elif operator == 'raw_numeric':
            feature_schema[group_name] = _generate_raw_numeric_columns(features)
    return feature_schema

def get_feature_schema(js):
    if not isinstance(js, dict):
        assert TypeError("js is not dict!")
    feature_schema = {}
    for group_name,v in js['data_schema']['feature_group'].items():
        dtype = str(v['dtype'])
        default_value = v['default_value']
        features = v['features']
        operator = v['operator']
        for key in features:
            if operator is not None and operator in ['pretrain']:
                dimension = v['dimension']
                feature_schema.update(
                    {key: tf.io.FixedLenFeature((dimension,), eval("tf.{}".format(dtype)), default_value=default_value)}
                )
            elif operator is not None and operator in ['hash_bucket', 'hash_bucket_idx']:
                shape = 1
                if isinstance(v['default_value'], list):
                    shape = len(v['default_value'])
                feature_schema.update(
                    {key: tf.io.FixedLenFeature([shape], eval("tf.{}".format(dtype)), default_value=default_value)}
                )
            else:
                feature_schema.update(
                    {key: tf.io.FixedLenFeature((1,), eval("tf.{}".format(dtype)), default_value=default_value)}
                )
    return feature_schema

def get_label_schema(js):
    if not isinstance(js, dict):
        assert TypeError("js is not dict!")
    labels = js['data_schema']['labels']
    label_schema = {}
    label_schema.update({
        name: tf.io.FixedLenFeature((1,), eval("tf.{}".format(config["dtype"])), default_value=config["default_value"])
        for name, config in labels.items()
    })
    return label_schema

def build_raw_serving_input_receiver_fn(js, exclude_feats = None):
    if not isinstance(js, dict):
        assert TypeError("feature_schemas is not dict!")
    if not isinstance(exclude_feats, list):
        assert TypeError("exclude_feats is not list!")
    features_placeholders = {}

    for group_name, v in js['data_schema']['feature_group'].items():
        dtype = str(v['dtype'])
        features = v['features']
        operator = v['operator']
        for key in features:
            if exclude_feats is not None and key in exclude_feats:
                continue
            if operator is not None and operator == 'pretrain':
                dimension = v['dimension']
                features_placeholders.update(
                    {key: tf.placeholder(dtype=eval("tf.{}".format(dtype)), shape=[None, dimension], name=key)}
                )
            elif operator is not None and operator == 'hash_bucket':
                if 'max_len' in v:
                    shape = v['max_len']
                    features_placeholders.update(
                        {key: tf.placeholder(dtype=eval("tf.{}".format(dtype)), shape=[None, shape], name=key)}
                    )
                else:
                    features_placeholders.update(
                        {key: tf.placeholder(dtype=eval("tf.{}".format(dtype)), shape=[None, ], name=key)}
                    )
            else:
                features_placeholders.update(
                    {key: tf.placeholder(dtype=eval("tf.{}".format(dtype)), shape=[None,], name=key)}
                )
    print(features_placeholders)
    return tf.estimator.export.build_raw_serving_input_receiver_fn(features_placeholders)

if __name__ == '__main__':
    pass