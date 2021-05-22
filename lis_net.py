import tensorflow as tf


def conv2d_batch_norm(input_layer, num_filters, kernel_size):
    out = tf.keras.layers.SeparableConv2D(
        filters=num_filters, kernel_size=kernel_size, padding='same'
    )(input_layer)
    out = tf.keras.layers.BatchNormalization(scale=False)(out)
    out = tf.keras.layers.Activation('relu')(out)
    return out


def lis_core(input_layer, num_filters):
    conv1x1 = conv2d_batch_norm(input_layer, num_filters, 1)
    conv3x3 = conv2d_batch_norm(conv1x1, num_filters, 3)
    conv5x5 = conv2d_batch_norm(conv3x3, num_filters, 3)

    out = tf.concat([conv1x1, conv3x3, conv5x5], -1)

    out = conv2d_batch_norm(out, num_filters, 1)

    out = tf.concat([input_layer, out], -1)
    out = tf.keras.layers.Activation('relu')(out)
    return out


def lis_block(input_layer, num_filters, number_of_lis_cores):
    out = conv2d_batch_norm(input_layer, num_filters, 1)

    for _ in range(0, number_of_lis_cores):
        out = lis_core(out, num_filters)

    out = conv2d_batch_norm(out, num_filters, 1)
    out = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                                    padding='same')(out)
    return out


def prepare_lis_net_model(
    input_shape,
    num_labels,
    base_param=48,
    number_of_lis_cores=[1, 2, 3, 4],
    growth_rate=[1, 2, 4, 8]
):
    inputs = tf.keras.layers.Input(input_shape)

    out = inputs
    for i, n_lis_cores in enumerate(number_of_lis_cores):
        number_of_filters = base_param * growth_rate[i]
        out = lis_block(out, number_of_filters, n_lis_cores)

    out = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                                    padding='same')(out)
    out = conv2d_batch_norm(out, base_param, 1)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(num_labels, activation='softmax')(out)

    return tf.keras.Model(inputs=inputs, outputs=out)
