# Pooling should be used as the last operation before softmax activation even
# though it isn't mentioned in the scientific paper about MatchboxNet
# AdaptiveAveragePooling1D(1) (from PyTorch) == GlobalAveragePooling1D()
# https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/matchboxnet_3x1x64_v2.yaml
# https://github.com/NVIDIA/NeMo/blob/0b36ea97277db6eb928f84f2c67d7ba499e10b6a/nemo/collections/asr/modules/conv_asr.py#L313

import tensorflow as tf


def sub_block(
    input_layer,
    channels,
    kernel_size,
    dropout_rate,
    stride=1,
    dilation=1,
    padding='same'
):
    out = tf.keras.layers.SeparableConv1D(
        filters=channels,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        dilation_rate=dilation
    )(input_layer)
    out = tf.keras.layers.BatchNormalization(scale=False)(out)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    return out


def block(input_layer, num_of_sub_blocks, channels, kernel_size, dropout_rate):
    out = input_layer
    for _ in range(num_of_sub_blocks - 1):
        out = sub_block(out, channels, kernel_size, dropout_rate, padding='same')

    resid = tf.keras.layers.Conv1D(filters=channels, kernel_size=1,
                                   padding='same')(input_layer)
    resid = tf.keras.layers.BatchNormalization(scale=False)(resid)

    out = tf.keras.layers.SeparableConv1D(
        filters=channels, kernel_size=kernel_size, padding='same'
    )(out)
    out = tf.keras.layers.BatchNormalization(scale=False)(out)

    out = out + resid

    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    return out


def prepare_matchbox_net_model(
    input_shape,
    num_labels,
    num_of_sub_blocks=2,
    kernel_sizes=[11, 13, 15, 17, 29],  # [prolog, blocks..., epilog]
    block_channels=64,
    prolog_epilog_channels=128,
    dropout_rate=0.2
):
    inputs = tf.keras.layers.Input(input_shape)
    inputs_reshape = tf.reshape(
        inputs, [-1, input_shape[0], input_shape[1] * input_shape[2]]
    )

    out = sub_block(
        inputs_reshape,
        prolog_epilog_channels,
        kernel_sizes[0],
        dropout_rate,
        stride=2,
        padding='same'
    )

    for i in kernel_sizes[1:-1]:
        out = block(out, num_of_sub_blocks, block_channels, i, dropout_rate)

    out = sub_block(
        out, prolog_epilog_channels, kernel_sizes[-1], dropout_rate, dilation=2
    )
    out = sub_block(out, prolog_epilog_channels, 1, dropout_rate)
    out = tf.keras.layers.Conv1D(filters=num_labels, kernel_size=1, padding='same')(out)
    
    out = tf.keras.layers.GlobalAveragePooling1D()(out)
    out = tf.keras.layers.Softmax(axis=-1)(out)

    return tf.keras.Model(inputs=inputs, outputs=out)
