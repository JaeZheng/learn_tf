#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/9/9 17:26
# @File    : seq2seq_train.py

import tensorflow as tf

MAX_LEN = 50
SOS_ID = 1


# 使用Dataset从一个文件中读取一个语言的数据
def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    # 根据空格将单词编号切分开并放入一个一维向量
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # 将字符串形式的单词编号转化为整数
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
    # 统计每个句子的单词数量，并与句子内容一起放入Dataset中
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset


# 从源语言文件和目标语言文件读取数据，并进行填充和batching操作
def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)

    dataset = tf.data.Dataset.zip((src_data, trg_data))

    # 删除内容为空的句子和长度过长的句子
    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        return tf.logical_and(src_len_ok, trg_len_ok)
    dataset = dataset.filter(FilterLength)

    # 制做编码器decoder的输入和输出
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))
    dataset = dataset.map(MakeTrgInput)
    dataset = dataset.shuffle(10000)

    # 规定填充后输出的数据维度
    padded_shapes = (
        (tf.TensorShape([None]),  # 源句子是长度未知的向量
         tf.TensorShape([])),     # 源句子长度是单个数字
        (tf.TensorShape([None]),  # 目标句子（解码器输入）是长度未知的向量
         tf.TensorShape([None]),  # 目标句子（解码器目标输出）是长度位置的向量
         tf.TensorShape([]))      # 目标句子长度是单个数字
    )
    # 调用padded_batch方法进行batching操作
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset


SRC_TRAIN_DATA = "./data/train.en"
TRG_TRAIN_DATA = "./data/train.zh"
CHECKPOINT_PATH = "./model/attention/attention_ckpt"
HIDDEN_SIZE = 1024               # LSTM的隐藏层节点数量
NUM_LAYERS = 2                   # 深层循环神经网络中LSTM结构的层数
SRC_VOCAB_SIZE = 10000           # 源语言词汇表大小
TRG_VOCAB_SIZE = 4000            # 目标语言词汇表大小
BATCH_SIZE = 100
NUM_EPOCH = 5
KEEP_PROB = 0.8
MAX_GRAD_NORM = 5
SHARE_EMB_AND_SOFTMAX = True


class NMTModel(object):
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构
        self.enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        self.enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
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

    # 在forward函数中定义模型的前向计算图
    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]

        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        # 使用bidirectional_dynamic_rnn构造编码器
        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
                self.enc_cell_fw, self.enc_cell_bw, src_emb, src_size, dtype=tf.float32)
            # 将两个LSTM的输出拼接为一个张量
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)
        # 构造解码器, 第一步的隐藏状态为编码器最后一层的隐藏状态
        with tf.variable_scope("decoder"):
            # 选择注意力权重的计算模型。
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                HIDDEN_SIZE, enc_outputs,
                memory_sequence_length=src_size
            )
            # 将解码器的循环神经网络self.dec_cell和注意力一起封装成更高层的循环神经网络
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.dec_cell, attention_mechanism,
                attention_layer_size=HIDDEN_SIZE
            )
            # 使用attention_cell和dynamic_rnn构造编码器
            dec_outputs, _ = tf.nn.dynamic_rnn(attention_cell, trg_emb, trg_size, dtype=tf.float32)

        # 计算器解码器每一步的perplexity
        output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label, [-1]),
                                                              logits=logits)

        # 计算平均损失时，需要将填充位置的权重设为0，以避免无效位置的预测干扰模型的训练
        label_weights = tf.sequence_mask(trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        trainanle_variables = tf.trainable_variables()

        grads = tf.gradients(cost/tf.to_float(batch_size), trainanle_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainanle_variables))
        return cost_per_token, train_op


def run_epoch(session, cost_op, train_op, saver, step):
    while True:
        try:
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                print("After %d steps, per token cost is %.3f" % (step, cost))
            if step % 200 == 0:
                saver.save(session, CHECKPOINT_PATH, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step


def main():
    initializer = tf.random_uniform_initializer()

    with tf.variable_scope("nmt_model", reuse=None, initializer=initializer):
        train_model = NMTModel()

    data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    cost_op, train_op = train_model.forward(src, src_size, trg_input, trg_label, trg_size)

    saver = tf.train.Saver()
    step = 0

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i+1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)


if __name__ == "__main__":
    main()