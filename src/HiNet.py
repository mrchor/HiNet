#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     HiNet
   Author :       mrchor
-------------------------------------------------
"""

import glob
import logging
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import random
import shutil
from utils.model_exporter import model_best_exporter
import tensorflow as tf
from utils.json_reader import load_json
from utils import schema_utils
from utils import file_gen
from tensorflow_estimator.python.estimator.canned import metric_keys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("job_name", 'hinet_train', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("num_threads", 64, "Number of threads")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 8, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 4096, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 100, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.05, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {SGD, Adam, Adagrad, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '128,64', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_string('train_data_path', '../data/', 'train data path')
tf.app.flags.DEFINE_string('eval_data_path', '../data/', 'evaluate data path')
tf.app.flags.DEFINE_string("model_dir", './model/hinet', "src check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", './model/hinet', "export servable src for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing src or not")
tf.app.flags.DEFINE_string("pos_weights", "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0", "positive weights")
tf.app.flags.DEFINE_integer("experts_num", 8, "expert nums")
tf.app.flags.DEFINE_integer("task_num", 12, "task nums")
tf.app.flags.DEFINE_integer('scenario_num', 6, 'scenario nums')
tf.app.flags.DEFINE_string("loss_weights", '1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0', "loss weights")
# tower units
tf.app.flags.DEFINE_string("tower_units", '128,64', "tower units")
tf.app.flags.DEFINE_string("tower_dropouts", '0.5,0.5', "tower dropout")
# scenario gate units
tf.app.flags.DEFINE_string("scenario_gate_units", '128,64,1', "scenario gate units")
# scenario SEI module setting
tf.app.flags.DEFINE_integer("scenario_shared_subexpert_nums", 5, "scenario shared sub expert nums")
tf.app.flags.DEFINE_string("scenario_shared_subexpert_units", '128,64', "scenario specific subexpert nums")
tf.app.flags.DEFINE_string("scenario_specific_subexpert_nums", '5,5,5,5,5,5,5,5,5', "scenario specific subexpert nums")
tf.app.flags.DEFINE_string("scenario_specific_subexpert_units", '128,64', "scenario specific subexpert units")
# CGC setting
tf.app.flags.DEFINE_string("exp_per_task", '2,2,2,2,2,2,2,2,2,2,2,2', "specific expert num")
tf.app.flags.DEFINE_integer("shared_num", '2', "shared expert num")
tf.app.flags.DEFINE_integer("level_number", '1', "cgc")

tf.app.flags.DEFINE_string("config_file_path", '../config/hinet_sample_schemas.json', "config path")

# log level
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(asctime)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# parse config file
config = load_json(json_file_path=FLAGS.config_file_path)
features_schemas = schema_utils.get_feature_schema(config)
feature_transforms = schema_utils.get_feature_transform(config)
label_schemas = schema_utils.get_label_schema(config)
print(feature_transforms)
fc = tf.feature_column

def input_fn(filenames, batch_size=32, num_epochs=None, perform_shuffle=False):
    """
    Args:
        filenames: TFRecord files
        batch_size: batch_size
        num_epochs: epoch
        perform_shuffle: is shuffle

    Returns:
        tensor
    """
    def _parse_fn(record):
        # feature parser
        features = tf.io.parse_single_example(record, features_schemas)
        labels = tf.io.parse_single_example(record, label_schemas)
        return features, labels

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TFRecordDataset(filenames).map(_parse_fn, num_parallel_calls=10).prefetch(
        1000000)  # multi-thread pre-process then prefetch

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)  # Batch size to use

    return dataset

def cgc_net(inputs, is_last, l2_reg, name):
    # inputs: [input_task1, input_task2 ... input_taskn, shared_input]
    inputs_final = []
    for input in inputs:
        input_shape = input.get_shape().as_list()
        inputs_final.append(tf.reshape(input, shape=[-1, 1, input_shape[1]]))
    expert_outputs = []
    exp_per_task = list(map(int, FLAGS.exp_per_task.strip().split(',')))
    deep_layers = list(map(int, FLAGS.deep_layers.strip().split(',')))
    # task-specific expert part
    for i in range(0, FLAGS.task_num):
        for j in range(0, exp_per_task[i]):
            inp = inputs_final[i]
            for unit in deep_layers:
                inp = tf.contrib.layers.fully_connected(inputs=inp, num_outputs=unit,
                                                        activation_fn=tf.nn.relu, \
                                                        weights_regularizer=l2_reg)
            expert_outputs.append(inp)  # None * 1 * 64
    # shared expert part
    for i in range(0, FLAGS.shared_num):
        inp = inputs_final[-1]
        for unit in deep_layers:
            inp = tf.contrib.layers.fully_connected(inputs=inp, num_outputs=unit,
                                                    activation_fn=tf.nn.relu, \
                                                    weights_regularizer=l2_reg)
        expert_outputs.append(inp)  # None * 1 * 64

    # shared gate
    outputs = []
    if is_last:
        for i in range(0, FLAGS.task_num):
            cur_expert_num = exp_per_task[i] + FLAGS.shared_num
            cur_gate = tf.contrib.layers.fully_connected(inputs=inputs[i], num_outputs=cur_expert_num,
                                                         activation_fn=tf.nn.relu, \
                                                         weights_regularizer=l2_reg)  # None * cur_expert_num
            cur_gate_shape = cur_gate.get_shape().as_list()
            cur_gate = tf.reshape(cur_gate, shape=[-1, cur_gate_shape[1], 1])
            cur_gate = tf.nn.softmax(cur_gate, axis=-1)
            # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
            cur_experts = expert_outputs[i * exp_per_task[i]:(i + 1) * exp_per_task[i]] + expert_outputs[
                                                                                          -int(FLAGS.shared_num):]
            expert_concat = tf.concat(cur_experts, axis=1)  # None * cur_expert_num * 64
            cur_gate_expert = tf.multiply(expert_concat, cur_gate)
            cur_gate_expert = tf.reduce_sum(cur_gate_expert, axis=1)  # None * 64
            outputs.append(cur_gate_expert)
    else:
        all_expert_num = FLAGS.shared_num
        for expert_num in exp_per_task:
            all_expert_num += expert_num
        for i in range(0, FLAGS.task_num + 1):
            cur_gate = tf.contrib.layers.fully_connected(inputs=inputs[i], num_outputs=all_expert_num,
                                                         activation_fn=tf.nn.relu, \
                                                         weights_regularizer=l2_reg)  # None * cur_expert_num
            cur_gate_shape = cur_gate.get_shape().as_list()
            cur_gate = tf.reshape(cur_gate, shape=[-1, cur_gate_shape[1], 1])
            cur_gate = tf.nn.softmax(cur_gate, axis=-1)
            # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
            cur_experts = expert_outputs
            expert_concat = tf.concat(cur_experts, axis=1)  # None * all_expert_num * 64
            cur_gate_expert = tf.multiply(expert_concat, cur_gate)
            cur_gate_expert = tf.reduce_sum(cur_gate_expert, axis=1)  # None * 64
            outputs.append(cur_gate_expert)

    return outputs

def subexpert_integration(input, mode, name, l2_reg, subexpert_nums = 5, subexpert_units = '128,64'):
    """
    subexpert integration module
    """
    subexpert_units = list(map(int, subexpert_units.split(',')))
    subexperts = []
    for j in range(subexpert_nums):
        subexpert = input
        for i in range(len(subexpert_units)):
            subexpert = tf.layers.dense(inputs=subexpert, units=subexpert_units[i],
                                                  activation=tf.nn.relu,
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                  bias_initializer=tf.zeros_initializer(),
                                                  name='subexpert_%s_%d_%d' % (name, j, i))
            if mode == tf.estimator.ModeKeys.TRAIN:
                subexpert = tf.nn.dropout(subexpert, keep_prob=0.5)
        subexperts.append(subexpert)
    subexperts = tf.concat([tf.expand_dims(se, axis=1) for se in subexperts], axis=1) # None * 5 * 64
    gate_network = tf.contrib.layers.fully_connected(
        inputs=input,
        num_outputs=subexpert_nums,
        activation_fn=tf.nn.relu, \
        weights_regularizer=l2_reg)
    gate_network_shape = gate_network.get_shape().as_list()
    gate_network = tf.nn.softmax(gate_network, axis=1)
    gate_network = tf.reshape(gate_network, shape=[-1, gate_network_shape[1], 1])  # None * 5 * 1
    output = tf.multiply(subexperts, gate_network) # None * 5 * 64
    output = tf.reduce_sum(output, axis=1) # None * 64
    return output

def model_fn(features, labels, mode, params):
    """bulid estimator model"""
    # ------hyperparameters----
    learning_rate = FLAGS.learning_rate
    l2_reg = tf.contrib.layers.l2_regularizer(FLAGS.l2_reg)

    # ------Input-------
    scenario_indicator = fc.input_layer(features=features,
                                            feature_columns=feature_transforms['scenario_indicator']['columns'])
    scenario_indicator_embedding = fc.input_layer(features=features,
                                        feature_columns=feature_transforms['scenario_indicator_embedding']['columns'])
    raw_features = fc.input_layer(features=features,
                                            feature_columns=feature_transforms['features']['columns'])

    input_features = tf.concat([raw_features], axis=-1)
    input_features = tf.layers.batch_normalization(input_features)

    # HiNet implement
    # scenario shared expert
    with tf.variable_scope('shared-expert-part'):
        scenario_shared_expert = subexpert_integration(input=input_features, mode=mode, name='scenario_shared_expert',
                                                    l2_reg=l2_reg, subexpert_nums=FLAGS.scenario_shared_subexpert_nums,
                                                    subexpert_units=FLAGS.scenario_shared_subexpert_units)

    # scenario extract module
    scenario_experts = []
    with tf.variable_scope('scenario-extract-module-part'):
        # init scenario expert
        scenario_specific_subexpert_nums = list(map(int, FLAGS.scenario_specific_subexpert_nums.split(',')))
        for j in range(FLAGS.scenario_num):
            scenario_expert = subexpert_integration(input=input_features, mode=mode, name='scenario_specific_expert_%d' % j,
                                                 l2_reg=l2_reg, subexpert_nums=scenario_specific_subexpert_nums[j],
                                                 subexpert_units=FLAGS.scenario_specific_subexpert_units)
            scenario_experts.append(tf.expand_dims(scenario_expert, axis=1))  # None * 1 * 64
        scenario_experts = tf.concat(scenario_experts, axis=1)  # None * 9 * 64
        # get current scenario expert
        cur_scenario_index = tf.one_hot(indices=tf.reshape(tf.cast(scenario_indicator, dtype=tf.int64), [-1, ]) - 1,
                                        depth=FLAGS.scenario_num)  # None * 9
        cur_scenario_index = tf.expand_dims(cur_scenario_index, axis=2)  # None * 9 * 1
        scenario_specific_expert = tf.multiply(scenario_experts, cur_scenario_index)  # None * 9 * 64
        scenario_specific_expert = tf.reduce_sum(scenario_specific_expert, axis=1, keepdims=True)  # None * 1 * 64

        scenario_expert_gate = scenario_indicator_embedding

        scenario_gate_units = list(map(int, FLAGS.scenario_gate_units.split(',')))  # 128, 64, 9
        scenario_gate_units.append(FLAGS.scenario_num)
        for i in range(len(scenario_gate_units)):
            scenario_expert_gate = tf.layers.dense(inputs=scenario_expert_gate, units=scenario_gate_units[i],
                                                   activation=tf.nn.relu,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                   bias_initializer=tf.zeros_initializer(),
                                                   name='scenario_expert_gate_%d' % i)  # None * 9

        all_scenario_index = tf.ones_like(cur_scenario_index, dtype=tf.int64)
        all_scenario_index = all_scenario_index - \
                             tf.expand_dims(tf.one_hot(
                                 indices=tf.reshape(tf.cast(scenario_indicator, dtype=tf.int64), [-1, ]) - 1,
                                 depth=FLAGS.scenario_num, dtype=tf.int64), axis=2)
        scenario_expert_gate = tf.expand_dims(scenario_expert_gate, axis=2)
        scenario_expert_gate = tf.nn.softmax(scenario_expert_gate, axis=1)  # None * 9 * 1
        scenario_transfer_expert = tf.multiply(scenario_expert_gate, scenario_experts)  # None * 9 * 64
        scenario_transfer_expert = tf.multiply(scenario_transfer_expert, tf.cast(all_scenario_index, dtype=tf.float32))
        scenario_transfer_expert = tf.reduce_sum(scenario_transfer_expert, axis=1)  # None * 64

    # concat scenario-specific expert, scenario-aware expert and scenario-shared expert
    scenario_specific_expert = tf.squeeze(scenario_specific_expert, axis=1)
    scenario_out_concat = tf.concat([scenario_transfer_expert, scenario_specific_expert, scenario_shared_expert], axis=-1)

    task_inputs = []
    for i in range(FLAGS.task_num + 1):
        task_inputs.append(scenario_out_concat)

    for i in range(FLAGS.level_number):
        if i == FLAGS.level_number - 1:  # final layer
            task_outputs = cgc_net(task_inputs, True, l2_reg, 'final-layer')
        else:
            task_inputs = cgc_net(task_inputs, False, l2_reg, 'not-final-layer')

    def tower(x, units_info, name):
        units = list(map(int, units_info.strip().split(',')))
        tower = x
        for i, unit in enumerate(units):
            tower = tf.layers.dense(inputs=tower, units=unit, activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer(), name='tower_%s_%d' % (name, i))
            # tower = activation_layers.activation_layer(inputs=tower, activation='evo_norm',
            #                                                prefix='evo_norm_%s_%d' % (name, i))
        return tower

    # scenario-a: 1; scenario-b: 2; scenario-c: 3; scenario-d:  4; scenario-e: 5; scenario-f: 6
    # scernario_a_is_order
    logits = []
    y_sigmoids = []
    y_scernario_a_is_order = task_outputs[0]
    y_scernario_a_is_order = tower(y_scernario_a_is_order, units_info=FLAGS.tower_units, name='y_scernario_a_is_order')
    y_scernario_a_is_order_ = tf.contrib.layers.fully_connected(inputs=y_scernario_a_is_order, num_outputs=1, activation_fn=None, \
                                               weights_regularizer=l2_reg,
                                               scope='scernario_a_is_order')
    y_scernario_a_is_order_ = tf.reshape(y_scernario_a_is_order_, [-1, ])
    logits.append(y_scernario_a_is_order_)
    y_scernario_a_is_order = tf.sigmoid(y_scernario_a_is_order_)
    y_sigmoids.append(y_scernario_a_is_order)

    import seaborn as sns
    sns.heatmap

    # scernario_a_is_click
    y_scernario_a_is_click = task_outputs[1]
    y_scernario_a_is_click = tower(y_scernario_a_is_click, units_info=FLAGS.tower_units, name='y_scernario_a_is_click')
    y_scernario_a_is_click_ = tf.contrib.layers.fully_connected(inputs=y_scernario_a_is_click, num_outputs=1, activation_fn=None, \
                                               weights_regularizer=l2_reg,
                                               scope='scernario_a_is_click')
    y_scernario_a_is_click_ = tf.reshape(y_scernario_a_is_click_, [-1, ])
    logits.append(y_scernario_a_is_click_)
    y_scernario_a_is_click = tf.sigmoid(y_scernario_a_is_click_)
    y_sigmoids.append(y_scernario_a_is_click)

    # scernario_b_is_order
    y_scernario_b_is_order = task_outputs[2]
    y_scernario_b_is_order = tower(y_scernario_b_is_order, units_info=FLAGS.tower_units, name='y_scernario_b_is_order')
    y_scernario_b_is_order_ = tf.contrib.layers.fully_connected(inputs=y_scernario_b_is_order, num_outputs=1, activation_fn=None, \
                                                          weights_regularizer=l2_reg,
                                                          scope='scernario_b_is_order')
    y_scernario_b_is_order_ = tf.reshape(y_scernario_b_is_order_, [-1, ])
    logits.append(y_scernario_b_is_order_)
    y_scernario_b_is_order = tf.sigmoid(y_scernario_b_is_order_)
    y_sigmoids.append(y_scernario_b_is_order)

    # scernario_b_is_click
    y_scernario_b_is_click = task_outputs[3]
    y_scernario_b_is_click = tower(y_scernario_b_is_click, units_info=FLAGS.tower_units, name='y_scernario_b_is_click')
    y_scernario_b_is_click_ = tf.contrib.layers.fully_connected(inputs=y_scernario_b_is_click, num_outputs=1, activation_fn=None, \
                                                          weights_regularizer=l2_reg,
                                                          scope='scernario_b_is_click')
    y_scernario_b_is_click_ = tf.reshape(y_scernario_b_is_click_, [-1, ])
    logits.append(y_scernario_b_is_click_)
    y_scernario_b_is_click = tf.sigmoid(y_scernario_b_is_click_)
    y_sigmoids.append(y_scernario_b_is_click)

    # scernario_c_is_order
    y_scernario_c_is_order = task_outputs[4]
    y_scernario_c_is_order = tower(y_scernario_c_is_order, units_info=FLAGS.tower_units, name='y_scernario_c_is_order')
    y_scernario_c_is_order_ = tf.contrib.layers.fully_connected(inputs=y_scernario_c_is_order, num_outputs=1, activation_fn=None, \
                                                           weights_regularizer=l2_reg,
                                                           scope='scernario_c_is_order')
    y_scernario_c_is_order_ = tf.reshape(y_scernario_c_is_order_, [-1, ])
    logits.append(y_scernario_c_is_order_)
    y_scernario_c_is_order = tf.sigmoid(y_scernario_c_is_order_)
    y_sigmoids.append(y_scernario_c_is_order)

    # scernario_c_is_click
    y_scernario_c_is_click = task_outputs[5]
    y_scernario_c_is_click = tower(y_scernario_c_is_click, units_info=FLAGS.tower_units, name='y_scernario_c_is_click')
    y_scernario_c_is_click_ = tf.contrib.layers.fully_connected(inputs=y_scernario_c_is_click, num_outputs=1, activation_fn=None, \
                                                           weights_regularizer=l2_reg,
                                                           scope='scernario_c_is_click')
    y_scernario_c_is_click_ = tf.reshape(y_scernario_c_is_click_, [-1, ])
    logits.append(y_scernario_c_is_click_)
    y_scernario_c_is_click = tf.sigmoid(y_scernario_c_is_click_)
    y_sigmoids.append(y_scernario_c_is_click)

    # scernario_d_is_order
    y_scernario_d_is_order = task_outputs[6]
    y_scernario_d_is_order = tower(y_scernario_d_is_order, units_info=FLAGS.tower_units, name='y_scernario_d_is_order')
    y_scernario_d_is_order_ = tf.contrib.layers.fully_connected(inputs=y_scernario_d_is_order, num_outputs=1, activation_fn=None, \
                                                           weights_regularizer=l2_reg,
                                                           scope='scernario_d_is_order')
    y_scernario_d_is_order_ = tf.reshape(y_scernario_d_is_order_, [-1, ])
    logits.append(y_scernario_d_is_order_)
    y_scernario_d_is_order = tf.sigmoid(y_scernario_d_is_order_)
    y_sigmoids.append(y_scernario_d_is_order)

    # scernario_d_is_click
    y_scernario_d_is_click = task_outputs[7]
    y_scernario_d_is_click = tower(y_scernario_d_is_click, units_info=FLAGS.tower_units, name='y_scernario_d_is_click')
    y_scernario_d_is_click_ = tf.contrib.layers.fully_connected(inputs=y_scernario_d_is_click, num_outputs=1, activation_fn=None, \
                                                           weights_regularizer=l2_reg,
                                                           scope='scernario_d_is_click')
    y_scernario_d_is_click_ = tf.reshape(y_scernario_d_is_click_, [-1, ])
    logits.append(y_scernario_d_is_click_)
    y_scernario_d_is_click = tf.sigmoid(y_scernario_d_is_click_)
    y_sigmoids.append(y_scernario_d_is_click)

    # scernario_e_is_order
    y_scernario_e_is_order = task_outputs[8]
    y_scernario_e_is_order = tower(y_scernario_e_is_order, units_info=FLAGS.tower_units, name='y_scernario_e_is_order')
    y_scernario_e_is_order_ = tf.contrib.layers.fully_connected(inputs=y_scernario_e_is_order, num_outputs=1, activation_fn=None, \
                                                          weights_regularizer=l2_reg,
                                                          scope='scernario_e_is_order')
    y_scernario_e_is_order_ = tf.reshape(y_scernario_e_is_order_, [-1, ])
    logits.append(y_scernario_e_is_order_)
    y_scernario_e_is_order = tf.sigmoid(y_scernario_e_is_order_)
    y_sigmoids.append(y_scernario_e_is_order)

    # scernario_e_is_click
    y_scernario_e_is_click = task_outputs[9]
    y_scernario_e_is_click = tower(y_scernario_e_is_click, units_info=FLAGS.tower_units, name='y_scernario_e_is_click')
    y_scernario_e_is_click_ = tf.contrib.layers.fully_connected(inputs=y_scernario_e_is_click, num_outputs=1, activation_fn=None, \
                                                          weights_regularizer=l2_reg,
                                                          scope='scernario_e_is_click')
    y_scernario_e_is_click_ = tf.reshape(y_scernario_e_is_click_, [-1, ])
    logits.append(y_scernario_e_is_click_)
    y_scernario_e_is_click = tf.sigmoid(y_scernario_e_is_click_)
    y_sigmoids.append(y_scernario_e_is_click)

    # scernario_f_is_order
    y_scernario_f_is_order = task_outputs[10]
    y_scernario_f_is_order = tower(y_scernario_f_is_order, units_info=FLAGS.tower_units, name='y_scernario_f_is_order')
    y_scernario_f_is_order_ = tf.contrib.layers.fully_connected(inputs=y_scernario_f_is_order, num_outputs=1, activation_fn=None, \
                                                           weights_regularizer=l2_reg,
                                                           scope='scernario_f_is_order')
    y_scernario_f_is_order_ = tf.reshape(y_scernario_f_is_order_, [-1, ])
    logits.append(y_scernario_f_is_order_)
    y_scernario_f_is_order = tf.sigmoid(y_scernario_f_is_order_)
    y_sigmoids.append(y_scernario_f_is_order)

    # scernario_f_is_click
    y_scernario_f_is_click = task_outputs[11]
    y_scernario_f_is_click = tower(y_scernario_f_is_click, units_info=FLAGS.tower_units, name='y_scernario_f_is_click')
    y_scernario_f_is_click_ = tf.contrib.layers.fully_connected(inputs=y_scernario_f_is_click, num_outputs=1, activation_fn=None, \
                                                           weights_regularizer=l2_reg,
                                                           scope='scernario_f_is_click')
    y_scernario_f_is_click_ = tf.reshape(y_scernario_f_is_click_, [-1, ])
    logits.append(y_scernario_f_is_click_)
    y_scernario_f_is_click = tf.sigmoid(y_scernario_f_is_click_)
    y_sigmoids.append(y_scernario_f_is_click)

    # 预测结果导出格式设置
    predictions = {
        "scernario_a_is_order": y_scernario_a_is_order,
        "scernario_a_is_click": y_scernario_a_is_click,
        "scernario_b_is_order": y_scernario_b_is_order,
        "scernario_b_is_click": y_scernario_b_is_click,
        "scernario_c_is_order": y_scernario_c_is_order,
        "scernario_c_is_click": y_scernario_c_is_click,
        "scernario_d_is_order": y_scernario_d_is_order,
        "scernario_d_is_click": y_scernario_d_is_click,
        "scernario_e_is_order": y_scernario_e_is_order,
        "scernario_e_is_click": y_scernario_e_is_click,
        "scernario_f_is_order": y_scernario_f_is_order,
        "scernario_f_is_click": y_scernario_f_is_click,
    }
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}
    # Estimator predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    label_scernario_a_is_order = tf.cast(tf.reshape(labels['scernario_a_is_order'], shape=[-1, ]), dtype=tf.float32)
    label_scernario_a_is_click = tf.cast(tf.reshape(labels['scernario_a_is_click'], shape=[-1, ]), dtype=tf.float32)
    label_scernario_b_is_order = tf.cast(tf.reshape(labels['scernario_b_is_order'], shape=[-1, ]), dtype=tf.float32)
    label_scernario_b_is_click = tf.cast(tf.reshape(labels['scernario_b_is_click'], shape=[-1, ]), dtype=tf.float32)
    label_scernario_c_is_order = tf.cast(tf.reshape(labels['scernario_c_is_order'], shape=[-1, ]), dtype=tf.float32)
    label_scernario_c_is_click = tf.cast(tf.reshape(labels['scernario_c_is_click'], shape=[-1, ]), dtype=tf.float32)
    label_scernario_d_is_order = tf.cast(tf.reshape(labels['scernario_d_is_order'], shape=[-1, ]), dtype=tf.float32)
    label_scernario_d_is_click = tf.cast(tf.reshape(labels['scernario_d_is_click'], shape=[-1, ]), dtype=tf.float32)
    label_scernario_e_is_order = tf.cast(tf.reshape(labels['scernario_e_is_order'], shape=[-1, ]), dtype=tf.float32)
    label_scernario_e_is_click = tf.cast(tf.reshape(labels['scernario_e_is_click'], shape=[-1, ]), dtype=tf.float32)
    label_scernario_f_is_order = tf.cast(tf.reshape(labels['scernario_f_is_order'], shape=[-1, ]), dtype=tf.float32)
    label_scernario_f_is_click = tf.cast(tf.reshape(labels['scernario_f_is_click'], shape=[-1, ]), dtype=tf.float32)

    loss_weights = list(map(float, FLAGS.loss_weights.strip().split(',')))
    pos_weights = list(map(float, FLAGS.pos_weights.strip().split(',')))
    labels = [label_scernario_a_is_order, label_scernario_a_is_click, label_scernario_b_is_order, label_scernario_b_is_click,label_scernario_c_is_order,
              label_scernario_c_is_click, label_scernario_d_is_order, label_scernario_d_is_click, label_scernario_e_is_order, label_scernario_e_is_click, label_scernario_f_is_order,
              label_scernario_f_is_click]
    loss = 0.0
    for i in range(FLAGS.task_num):
        loss += loss_weights[i] * tf.losses.sigmoid_cross_entropy(multi_class_labels=labels[i],
                                                             logits=logits[i],
                                                             weights=tf.add(
                                                                 labels[i] * (pos_weights[i] - 1.0),
                                                                 tf.ones_like(labels[i])))

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {}

    # scenario-a；
    scernario_a_is_order_idx = tf.where(tf.equal(tf.reshape(scenario_indicator, [-1, ]), 1))
    label_scernario_a_is_order = tf.gather_nd(label_scernario_a_is_order, scernario_a_is_order_idx)
    y_scernario_a_is_order = tf.gather_nd(y_scernario_a_is_order, scernario_a_is_order_idx)
    eval_metric_ops['auc_scernario_a_is_order'] = tf.metrics.auc(label_scernario_a_is_order, y_scernario_a_is_order)

    scernario_a_is_click_idx = tf.where(tf.equal(tf.reshape(scenario_indicator, [-1, ]), 1))
    label_scernario_a_is_click = tf.gather_nd(label_scernario_a_is_click, scernario_a_is_click_idx)
    y_scernario_a_is_click = tf.gather_nd(y_scernario_a_is_click, scernario_a_is_click_idx)
    eval_metric_ops['auc_scernario_a_is_click'] = tf.metrics.auc(label_scernario_a_is_click, y_scernario_a_is_click)

    # scenario-b；
    scernario_b_is_order_idx = tf.where(tf.equal(tf.reshape(scenario_indicator, [-1, ]), 2))
    label_scernario_b_is_order = tf.gather_nd(label_scernario_b_is_order, scernario_b_is_order_idx)
    y_scernario_b_is_order = tf.gather_nd(y_scernario_b_is_order, scernario_b_is_order_idx)
    eval_metric_ops['auc_scernario_b_is_order'] = tf.metrics.auc(label_scernario_b_is_order, y_scernario_b_is_order)

    scernario_b_is_click_idx = tf.where(tf.equal(tf.reshape(scenario_indicator, [-1, ]), 2))
    label_scernario_b_is_click = tf.gather_nd(label_scernario_b_is_click, scernario_b_is_click_idx)
    y_scernario_b_is_click = tf.gather_nd(y_scernario_b_is_click, scernario_b_is_click_idx)
    eval_metric_ops['auc_scernario_b_is_click'] = tf.metrics.auc(label_scernario_b_is_click, y_scernario_b_is_click)

    # scenario-c；
    scernario_c_is_order_idx = tf.where(tf.equal(tf.reshape(scenario_indicator, [-1, ]), 3))
    label_scernario_c_is_order = tf.gather_nd(label_scernario_c_is_order, scernario_c_is_order_idx)
    y_scernario_c_is_order = tf.gather_nd(y_scernario_c_is_order, scernario_c_is_order_idx)
    eval_metric_ops['auc_scernario_c_is_order'] = tf.metrics.auc(label_scernario_c_is_order, y_scernario_c_is_order)

    scernario_c_is_click_idx = tf.where(tf.equal(tf.reshape(scenario_indicator, [-1, ]), 3))
    label_scernario_c_is_click = tf.gather_nd(label_scernario_c_is_click, scernario_c_is_click_idx)
    y_scernario_c_is_click = tf.gather_nd(y_scernario_c_is_click, scernario_c_is_click_idx)
    eval_metric_ops['auc_scernario_c_is_click'] = tf.metrics.auc(label_scernario_c_is_click, y_scernario_c_is_click)

    # scenario-d；
    scernario_d_is_order_idx = tf.where(tf.equal(tf.reshape(scenario_indicator, [-1, ]), 4))
    label_scernario_d_is_order = tf.gather_nd(label_scernario_d_is_order, scernario_d_is_order_idx)
    y_scernario_d_is_order = tf.gather_nd(y_scernario_d_is_order, scernario_d_is_order_idx)
    eval_metric_ops['auc_scernario_d_is_order'] = tf.metrics.auc(label_scernario_d_is_order, y_scernario_d_is_order)

    scernario_d_is_click_idx = tf.where(tf.equal(tf.reshape(scenario_indicator, [-1, ]), 4))
    label_scernario_d_is_click = tf.gather_nd(label_scernario_d_is_click, scernario_d_is_click_idx)
    y_scernario_d_is_click = tf.gather_nd(y_scernario_d_is_click, scernario_d_is_click_idx)
    eval_metric_ops['auc_scernario_d_is_click'] = tf.metrics.auc(label_scernario_d_is_click, y_scernario_d_is_click)

    # scenario-e；
    scernario_e_is_order_idx = tf.where(tf.equal(tf.reshape(scenario_indicator, [-1, ]), 5))
    label_scernario_e_is_order = tf.gather_nd(label_scernario_e_is_order, scernario_e_is_order_idx)
    y_scernario_e_is_order = tf.gather_nd(y_scernario_e_is_order, scernario_e_is_order_idx)
    eval_metric_ops['auc_scernario_e_is_order'] = tf.metrics.auc(label_scernario_e_is_order, y_scernario_e_is_order)

    scernario_e_is_click_idx = tf.where(tf.equal(tf.reshape(scenario_indicator, [-1, ]), 5))
    label_scernario_e_is_click = tf.gather_nd(label_scernario_e_is_click, scernario_e_is_click_idx)
    y_scernario_e_is_click = tf.gather_nd(y_scernario_e_is_click, scernario_e_is_click_idx)
    eval_metric_ops['auc_scernario_e_is_click'] = tf.metrics.auc(label_scernario_e_is_click, y_scernario_e_is_click)

    # scenario-f；
    scernario_f_is_order_idx = tf.where(tf.equal(tf.reshape(scenario_indicator, [-1, ]), 6))
    label_scernario_f_is_order = tf.gather_nd(label_scernario_f_is_order, scernario_f_is_order_idx)
    y_scernario_f_is_order = tf.gather_nd(y_scernario_f_is_order, scernario_f_is_order_idx)
    eval_metric_ops['auc_scernario_f_is_order'] = tf.metrics.auc(label_scernario_f_is_order, y_scernario_f_is_order)

    scernario_f_is_click_idx = tf.where(tf.equal(tf.reshape(scenario_indicator, [-1, ]), 6))
    label_scernario_f_is_click = tf.gather_nd(label_scernario_f_is_click, scernario_f_is_click_idx)
    y_scernario_f_is_click = tf.gather_nd(y_scernario_f_is_click, scernario_f_is_click_idx)
    eval_metric_ops['auc_scernario_f_is_click'] = tf.metrics.auc(label_scernario_f_is_click, y_scernario_f_is_click)

    # 以点评秒杀下单为模型评估基准
    eval_metric_ops['auc'] = tf.metrics.auc(label_scernario_a_is_order, y_scernario_a_is_order)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    # ------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-6)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)


def main(_):
    tr_files = file_gen.get_local_path(base_path=FLAGS.train_data_path)
    random.shuffle(tr_files)
    print("tr_files:", tr_files)
    va_files = file_gen.get_local_path(base_path=FLAGS.eval_data_path)
    print("va_files:", va_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing src cleaned at %s" % FLAGS.model_dir)

    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    config_proto = tf.ConfigProto(allow_soft_placement=True,
                                  intra_op_parallelism_threads=0,
                                  inter_op_parallelism_threads=0,
                                  log_device_placement=False,
                                  )
    run_config = tf.estimator.RunConfig(train_distribute=strategy, eval_distribute=strategy, session_config=config_proto,
                                    log_step_count_steps=FLAGS.log_steps, save_checkpoints_steps=FLAGS.log_steps,
                                    save_summary_steps=FLAGS.log_steps, tf_random_seed=2021)

    HiNet = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, config=run_config)

    serving_input_receiver_fn = schema_utils.build_raw_serving_input_receiver_fn(config)

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs,
                                      batch_size=FLAGS.batch_size))

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size),
            steps=None,
            exporters=[model_best_exporter(FLAGS.job_name, serving_input_receiver_fn, exports_to_keep=1,
                                           metric_key=metric_keys.MetricKeys.AUC, big_better=False)],
            start_delay_secs=10, throttle_secs=10
        )
        tf.estimator.train_and_evaluate(HiNet, train_spec, eval_spec)
    elif FLAGS.task_type == 'eval':
        HiNet.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
