"""
Provide the images both in low and high resolution,
the low res images are used as input1, saliency1 is used to sample the high res images to provide input2.
The dataset should be saved in the TF-record.
"""


import tensorflow as tf


def image_dataaugmentation(image):

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image


def image_standardization(image):
    image = image / 255.

    return image

def train_parser(record):

    features = tf.parse_single_example(
        record,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/depth': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            # 'image': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    label = tf.cast(features['image/class/label'], tf.int64)
    height = tf.cast(features['image/height'], tf.int64)
    width = tf.cast(features['image/width'], tf.int64)
    depth = tf.cast(features['image/depth'], tf.int64)

    image = tf.to_float(image)
    image = tf.reshape(image, [320, 320, 3])
    image = image_standardization(image)
    label = tf.one_hot(label, 3)

    high_res_image_batch = image_dataaugmentation(image)
    low_res_image_batch = tf.image.resize_images(high_res_image_batch, (128, 128))


    return high_res_image_batch, low_res_image_batch, label


def test_parser(record):

    features = tf.parse_single_example(
        record,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/depth': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            # 'image': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    label = tf.cast(features['image/class/label'], tf.int64)
    height = tf.cast(features['image/height'], tf.int64)
    width = tf.cast(features['image/width'], tf.int64)
    depth = tf.cast(features['image/depth'], tf.int64)

    image = tf.to_float(image)
    image = tf.reshape(image, [320, 320, 3])
    high_res_image_batch = image_standardization(image)
    label = tf.one_hot(label, 3)

    low_res_image_batch = tf.image.resize_images(high_res_image_batch, (128, 128))


    return high_res_image_batch, low_res_image_batch, label



####################### Train_Dataset ########################

def data_loader(tfrecord_path, parser, batch_size, epoch, type, buffer_size = 100):

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parser)
    if 'train' in type:
        dataset = dataset.shuffle(buffer_size)
        print('train_dataset:need shuffle!')
    elif 'test' in type:
        print('test_dataset:do not need shuffle!')

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epoch)
    iterator = dataset.make_one_shot_iterator()
    high_res_image, low_res_image, label = iterator.get_next()

    return high_res_image, low_res_image, label



# train_step_per_epoch = int(np.ceil(1449 / BATCH_SIZE))


####################### Test_Dataset ########################

# test_dataset = tf.data.TFRecordDataset('tfrecord/test')
# test_dataset = test_dataset.map(test_parser)
# test_dataset = test_dataset.shuffle(buffer_size = 100)
# test_dataset = test_dataset.batch(BATCH_SIZE)
# test_dataset = test_dataset.repeat(EPOCH)
# test_iterator = test_dataset.make_one_shot_iterator()
# test_high_res_image, test_low_res_image, test_label = test_iterator.get_next()

# test_step_per_epoch = int(np.ceil(363 / BATCH_SIZE))







