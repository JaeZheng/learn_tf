#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/9 20:44
# @File    : seq2seq_test.py

import tensorflow as tf
import codecs

CHECKPOINT_PATH = "./model/seq2seq_ckpt-9000"
SRC_VOCAB_FILE = "./data/en.vocab"
TRG_VOCAB_FILE = "./data/zh.vocab"

HIDDEN_SIZE = 1024               # LSTM的隐藏层节点数量
NUM_LAYERS = 2                   # 深层循环神经网络中LSTM结构的层数
SRC_VOCAB_SIZE = 10000           # 源语言词汇表大小
TRG_VOCAB_SIZE = 4000            # 目标语言词汇表大小
SHARE_EMB_AND_SOFTMAX = True

SOS_ID = 1
EOS_ID = 2


class NMTModel(object):
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)]
        )
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)]
        )

        self.src_embedding = tf.get_variable("src_emb", [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable("trg_emb", [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable("softmax_weight", [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable("softmax_bias", [TRG_VOCAB_SIZE])

    def inference(self, src_input):
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        # 使用dynamic_rnn构造编码器,这一步与训练时相同
        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(self.enc_cell, src_emb, src_size, dtype=tf.float32)
        # 设置解码的最大步数，防止无限循环
        MAX_DEC_LEN = 100

        with tf.variable_scope("decoder/rnn/multi_rnn_cell"):
            # 使用一个变长的TensorArray来存储生成的句子
            init_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)
            # 填入第一个词<SOS>作为解码器的输入
            init_array = init_array.write(0, SOS_ID)
            # 构建初始的循环状态。
            init_loop_var = (enc_state, init_array, 0)

            # tf.while_loop的循环条件
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), EOS_ID),
                    tf.less(step, MAX_DEC_LEN-1)
                ))

            def loop_body(state, trg_ids, step):
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)
                # 这里不是用dynamic_rnn, 而是直接调用dec_cell向前计算一步
                dec_outputs, next_state = self.dec_cell.call(state=state, inputs=trg_emb)
                # 计算每个可能输出的单词对应的logit，并选取logit最大的单词作为这一步的输出
                output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
                logits = (tf.matmul(output, self.softmax_weight) + self.softmax_bias)
                nex_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                # 将这一步输出的单词写入循环状态的trg_ids中
                trg_ids = trg_ids.write(step+1, nex_id[0])
                return next_state, trg_ids, step+1

            # 执行tf.while_loop，并返回最终状态
            state, trg_ids, step = tf.while_loop(continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()


def get_sentence(word_sequence, vocab_file):
    with codecs.open(vocab_file, 'r', 'utf-8') as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]
    id_to_word = {v: k for (k, v) in zip(vocab, range(len(vocab)))}
    sentence = ""
    for word in word_sequence:
        sentence += " " + id_to_word[word]
    return sentence


def get_word_sequence(sentence, vocab_file):
    with codecs.open(vocab_file, 'r', 'utf-8') as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]
    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}
    sentence = sentence.split(" ")
    word_sequence = []
    for word in sentence:
        word_sequence.append(word_to_id[word])
    return word_sequence


def main():
    with tf.variable_scope("mmt_model", reuse=None):
        model = NMTModel()
    # test_sentence = [90, 13, 9, 689, 4, 2]
    test_sentence = "This is a test . <eos>"
    test_sentence = get_word_sequence(test_sentence, SRC_VOCAB_FILE)
    print(test_sentence)
    src_sentence = get_sentence(test_sentence, SRC_VOCAB_FILE)
    print(src_sentence)
    output_op = model.inference(test_sentence)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, CHECKPOINT_PATH)
    output = sess.run(output_op)
    trg_sentence = get_sentence(output, TRG_VOCAB_FILE)
    print(trg_sentence)
    sess.close()


if __name__ == '__main__':
    main()


