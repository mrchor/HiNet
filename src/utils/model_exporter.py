# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model_exporter
   Author :       mrchor
-------------------------------------------------
"""
import tensorflow as tf
from tensorflow.python.estimator.canned import metric_keys

# export best model
def model_best_exporter(model_name, serving_input_receiver_fn, assets_extra=None, exports_to_keep=1, metric_key=metric_keys.MetricKeys.LOSS, big_better=False):
    def compare(best_eval_result, current_eval_result):
        if not best_eval_result or metric_key not in best_eval_result:
            raise ValueError(
                'best_eval_result cannot be empty or no loss is found in it.')

        if not current_eval_result or metric_key not in current_eval_result:
            raise ValueError(
                'current_eval_result cannot be empty or no loss is found in it.')

        if big_better:
            return best_eval_result[metric_key] > current_eval_result[metric_key]
        else:
            return best_eval_result[metric_key] < current_eval_result[metric_key]

    return tf.estimator.BestExporter(name=model_name,
                                     serving_input_receiver_fn=serving_input_receiver_fn,
                                     compare_fn=compare,
                                     assets_extra=assets_extra,
                                     exports_to_keep=exports_to_keep)
