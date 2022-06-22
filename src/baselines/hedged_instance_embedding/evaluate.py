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


parser = argparse.ArgumentParser()

parser.add_argument('-lr', type=float, default=0.0001, help="learning rate")
parser.add_argument('-epoch', type=int, default=300, help="epoch")
parser.add_argument('-beta', type=float, default=0.00001, help="beta")
parser.add_argument('-samples', type=int, default=100, help="sampling")
parser.add_argument('-n_samples', type=int, default=50, help="train or test images")
parser.add_argument('-d_output', type=int, default=2, help="dimension of model ouput")
parser.add_argument('-batch_size', type=int, default=10000, help="dimension of model ouput")
parser.add_argument('-n_filter', type=int, default=3, help="dimension of model ouput")
parser.add_argument('-model', type=str, default="point", help="training model")
parser.add_argument('-testpath', type=str, default="test.tfrecord", help="test path")
parser.add_argument('-tfrecordpath', type=str, default="train.tfrecord", help="dimension of model ouput")
parser.add_argument('-actv', type=str, default="relu", help="dimension of model ouput")


args = parser.parse_args()

if __name__ == "__main__":
    tf.enable_eager_execution()

    dataloader = Dataloader(args)
    point_model_path = "{}_l_{}_b_{}_f_{}.h5".format('point', args.lr, 64, args.n_filter)
    hib_model_path = "{}_l_{}_b_{}_f_{}.h5".format('hib', args.lr, 64, args.n_filter)

    test_dataset = dataloader.create_dataset_test()
    test = iter(test_dataset).__next__()

    point = PointEmbedding(args)
    point_model = point.build()
    point_model.load_weights(point_model_path)
    
    X, Y  = test
    
    print(Y.shape)
    print(X.shape)

    print('---------------')
    print(type(test_dataset))

    hib = HedgedInstanceEmbedding(args)
    hib_model = hib.build()
    hib_model.load_weights(hib_model_path)

    print("Evaluate on test data")
    # results = model.evaluate(x_test, y_test, batch_size=128)
    # print("test loss, test acc:", results)