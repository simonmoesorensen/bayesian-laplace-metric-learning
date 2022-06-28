import numpy as np
import argparse
import tensorflow as tf
from dataloader import Dataloader
from tensorflow.python.keras import datasets
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input
from model import PointEmbedding, HedgedInstanceEmbedding
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.patches import Ellipse
from sklearn.preprocessing import normalize, MinMaxScaler

tf.enable_eager_execution()

parser = argparse.ArgumentParser()

parser.add_argument('-lr', type=float, default=0.0001, help="learning rate")
parser.add_argument('-epoch', type=int, default=300, help="epoch")
parser.add_argument('-beta', type=float, default=0.00001, help="beta")
parser.add_argument('-samples', type=int, default=100, help="sampling")
parser.add_argument('-n_samples', type=int, default=50, help="train or test images")
parser.add_argument('-d_output', type=int, default=2, help="dimension of model ouput")
parser.add_argument('-batch_size', type=int, default=20, help="dimension of model ouput")
parser.add_argument('-n_filter', type=int, default=3, help="dimension of model ouput")
parser.add_argument('-model', type=str, default="point", help="training model")
parser.add_argument('-testpath', type=str, default="test.tfrecord", help="test path")
parser.add_argument('-tfrecordpath', type=str, default="train.tfrecord", help="dimension of model ouput")
parser.add_argument('-actv', type=str, default="relu", help="dimension of model ouput")


args = parser.parse_args()

dataloader = Dataloader(args)
scaler = MinMaxScaler()

if __name__ == "__main__":
    point_model_path = "{}_l_{}_b_{}_f_{}.h5".format('point', args.lr, 64, args.n_filter)
    hib_model_path = "{}_l_{}_b_{}_f_{}.h5".format('hib', args.lr, 64, args.n_filter)

    test_dataset = dataloader.create_dataset_test()
    size = 0
    # Point Embedding model
    point = PointEmbedding(args)
    point_model = point.build()
    point_model.load_weights(point_model_path)


    test = np.zeros((64, 4, 28, 28), dtype=np.float32)
    output1 = point_model.layers[-3].output # z1
    output2 = point_model.layers[-2].output # z2
    # output: z1, z2
    point_model = Model(point_model.input, [output1, output2])

    # HIB model
    hib = HedgedInstanceEmbedding(args)
    hib_model = hib.build()
    hib_model.load_weights(hib_model_path)

    output1, output2 = hib_model.layers[-1].input

    # output: z1, z2
    hib_model = Model(hib_model.input, [output1, output2])

    test = iter(test_dataset).__next__()
    
    print('------')
    print(type(test))
    print(len(test))

    x, y = test
    y = y.numpy()

    h_1,h_2 = hib_model(x)

    print(np.array(h_1).shape)
    print(np.array(h_2).shape)

    h_1_m = np.mean(h_1, axis=-2)
    h_1_v = np.var(h_1, axis=-2)


    scaler = MinMaxScaler()
    h_1_m = scaler.fit_transform(h_1_m)
    scaler = MinMaxScaler()
    h_1_v = scaler.fit_transform(h_1_v)

    h_2_m = np.mean(h_2, axis=-2)
    h_2_v = np.var(h_2, axis=-2)

    scaler = MinMaxScaler()
    h_2_m = scaler.fit_transform(h_2_m)
    scaler = MinMaxScaler()
    h_2_v = scaler.fit_transform(h_2_v)

    print('h_1', h_1_m[0], h_1_v[0])
    print('h_2', h_2_m[0], h_2_v[0])

    print('---')

    print(y[0][0])
    print(y[0][1])
    print(y[0][2])
    print(y[0][3])
