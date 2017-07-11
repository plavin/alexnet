from collections import OrderedDict
from functools import partial
import layers
import arrayfire as af

class Afnet:
    def __init__(self, _params, _layers = OrderedDict(), _data = None):
        self.params = _params
        self.layers = _layers
        self.data   = _data

    def add_conv_layer(self, name, wx, wy, sx = 1, sy = 1, px = 0, py = 0, groups = 1):
        if name in self.layers:
            print "Error: layer name already in use"
            return

        self.layers[name] = partial(layers.conv, wx=wx, wy=wy, sx=sx, sy=sy,
                                    px=px, py=py, groups=groups)

    def add_relu_layer(self, name):
        if name in self.layers:
            print "Error: layer name already in use"
            return

        self.layers[name] = layers.relu

    def add_lrn_layer(self, name, size = 5, alpha = 1, beta = 5):
        if name in self.layers:
            print "Error: layer name already in use"
            return

        self.layers[name] = partial(layers.lrn, size=size, alpha=alpha, beta=beta)

    def add_pool_layer(self, name, w, s = 1):
        if name in self.layers:
            print "Error: layer name already in use"
            return

        self.layers[name] = partial(layers.pool, w=w, s=s)

    def add_fc_layer(self, name):
        if name in self.layers:
            print "Error: layer name already in use"
            return

        self.layers[name] = partial(layers.fc)

    def add_softmax_layer(self, name):
        if name in self.layers:
            print "Error: layer name already in use"
            return

        self.layers[name] = partial(layers.softmax)

    

    def forward(self, end):
        if len(self.layers) is 0:
            return self.data
        out = af.Array()
        data = self.data
        for key in self.layers.viewkeys():
            if key in self.params:
                data = self.layers[key](image = data,
                                        weights = self.params[key]['weights'],
                                        biases = self.params[key]['biases'])
            else:
                data = self.layers[key](image = data)

            if key is end:
                return data
        return data
