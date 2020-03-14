import json
import os
import urllib.request

import pandas as pd
import tensorflow as tf

import gdown

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
DATA_DIR = ROOT_DIR + 'data/'


dataset_url = 'https://drive.google.com/uc?id=1t_x0kzRufbaqyJ3dGTQK-QJO9TFfOu2g'

def _int64_feature(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _get_description(x):
    if x in (6, 9):
        # c6 and c9 are varlen
        return tf.io.VarLenFeature(tf.int64)
    else:
        # All other features are fixed len (one)
        return tf.io.FixedLenFeature([], tf.int64, default_value=0)


_feature_description = {'c%s' % x: _get_description(x) for x in range(10)}
_feature_description['l'] = _get_description('l')


def _parse(example_proto):
    return tf.io.parse_single_example(example_proto, _feature_description)


def _serialize(x):
    parsed = json.loads(x.numpy())
    feature = {k: _int64_feature([v] if k == 'l' else v) for k, v in parsed.items()}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def load_dataset():
    """Load and return the dataset
    Returns
    -------
    data : TODO
    --------
    """
    destination = os.path.join(DATA_DIR, 'dataset.jsons.gz')
    gdown.cached_download(dataset_url, destination, md5='a8f860b2dc400e14d4e7083775c1308a', quiet=False)

    if not os.path.isfile(destination):
        # Data not cached, download
        urllib.request.urlretrieve(dataset_url, destination)

    source_file = tf.constant([destination])

    dataset = tf.data.TextLineDataset(filenames=source_file, compression_type='GZIP')
    dataset = dataset.map(lambda x: tf.py_function(_serialize, [x], [tf.string]))
    dataset = dataset.map(_parse)

    return dataset


def split(train_size=800000, validation_size=100000):
    full_dataset = load_dataset()
    train_dataset = full_dataset.take(train_size)
    not_train_dataset = full_dataset.skip(train_size)
    val_dataset = not_train_dataset.take(validation_size)
    test_dataset = not_train_dataset.skip(validation_size)
    return train_dataset, val_dataset, test_dataset


def to_pandas_df(ds, n=1000):    
    ds = ds.batch(n).take(1)
    entire_dataset = tf.data.experimental.get_single_element(ds)

    def to_series(k, v):
        if k in ('c6', 'c9'):
            return pd.Series(tf.RaggedTensor.from_sparse(v).to_list(), dtype='object')
        else:
            return pd.Series(v.numpy(), dtype=pd.Int64Dtype())
	
    return pd.DataFrame({k: to_series(k, v) for k, v in entire_dataset.items()})


if __name__ == '__main__':
    ds = load_dataset()
    df = to_pandas_df(ds)
    print(df.head())
