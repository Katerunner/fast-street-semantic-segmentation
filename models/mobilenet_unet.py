import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


def create_model(input_shape=(128, 128, 3), num_classes=3, down_stack_trainable=False, encoder_name='MobileNetV2'):
    if encoder_name == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)
    elif encoder_name == 'MobileNetV3Small':
        base_model = tf.keras.applications.MobileNetV3Small(input_shape=input_shape, include_top=False)
    elif encoder_name == 'MobileNetV3Large':
        base_model = tf.keras.applications.MobileNetV3Large(input_shape=input_shape, include_top=False)
    elif encoder_name == 'MobileNet':
        base_model = tf.keras.applications.MobileNet(input_shape=input_shape, include_top=False)
    elif encoder_name == 'NASNetMobile':
        base_model = tf.keras.applications.NASNetMobile(input_shape=input_shape, include_top=False)
    elif encoder_name == 'ConvNeXtTiny':
        base_model = tf.keras.applications.ConvNeXtTiny(input_shape=input_shape, include_top=False)
    elif encoder_name == 'ConvNeXtSmall':
        base_model = tf.keras.applications.ConvNeXtSmall(input_shape=input_shape, include_top=False)
    elif encoder_name == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(input_shape=input_shape, include_top=False)
    elif encoder_name == 'EfficientNetV2S':
        base_model = tf.keras.applications.EfficientNetV2S(input_shape=input_shape, include_top=False)
    elif encoder_name == 'ResNetRS101':
        base_model = tf.keras.applications.ResNetRS101(input_shape=input_shape, include_top=False)
    else:
        raise ValueError("Invalid encoder name")

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = down_stack_trainable

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]

    model = unet_model(
            input_shape=input_shape,
            output_channels=num_classes,
            down_stack=down_stack,
            up_stack=up_stack)

    return model


def unet_model(input_shape: tuple, output_channels: int, down_stack, up_stack: list):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Down sampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Sampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
            filters=output_channels, kernel_size=3, strides=2,
            padding='same')  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
