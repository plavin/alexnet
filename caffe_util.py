import collections
import arrayfire as af
import caffe
import numpy as np
import os
from afnet import Afnet

#In arrayfire: af_params[layer]['weights'][a,b,c,d] is element a, b from channel c, input d
#In caffe this corresponds to net.params[key][0].data[d][c][a][b]

def load_caffe(deploy_prototxt, model):
    return caffe.Net(deploy_prototxt, model, caffe.TEST)


#caffe init materials from http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html
def load_image(caffe_net, image):
    tmp = af.load_image(image, is_color=True)
    af_image = af.resize(tmp, odim0=227, odim1=227)
    caffe_image = af_image.__array__()
    py_caffe = '/home/aatish/caffe/python/'

    t = caffe.io.Transformer({'data': caffe_net.blobs['data'].data.shape})
    t.set_mean('data', np.load(py_caffe + 'caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    t.set_transpose('data', (2,0,1))
    t.set_channel_swap('data', (2,1,0))
    t.set_raw_scale('data', 255.0)

    caffe_image = t.preprocess('data', af_image.__array__()).reshape(1, 3, 227, 227)
    caffe_net.blobs['data'].reshape(1, 3, 227, 227)
    caffe_net.blobs['data'].data[...] = caffe_image

    af_image = af.np_to_af_array(caffe_image)
    af_image = af.reorder(af_image, 2, 3, 1, 0)

    return af_image

def caffe_params_to_af_params(net):
    af_params = collections.OrderedDict()

    for key in net.params.viewkeys():
        af_params[key] = collections.OrderedDict()
        af_params[key]['weights'] = af.reorder(af.np_to_af_array(net.params[key][0].data), 2, 3, 1, 0)
        af_params[key]['biases']  = af.np_to_af_array(net.params[key][1].data)

    return af_params

def caffe_layers_to_af_layers(net):
    #TODO: this should be done automatically
    return

def caffe_to_af(net):
    af_params = caffe_params_to_af_params(net)
    #af_layers = caffe_layers_to_af_layers(net)
    return Afnet(af_params)

def compare_output(af_, caffe):
    print "[0,0,0,0]"
    print "caffe", caffe[0][0][0][0]
    print "af", af_[0,0,0,0].to_list()[0]

    print "[0,1,0,0]"
    print "caffe", caffe[0][0][0][1]
    print "af", af_[0,1,0,0].to_list()[0]

    print "[1,0,0,0]"
    print "caffe", caffe[0][0][1][0]
    print "af", af_[1,0,0,0].to_list()[0]

    print "[0,0,1,0]"
    print "caffe", caffe[0][1][0][0]
    print "af", af_[0,0,1,0].to_list()[0]

def num_filters(layer):
    return layer.dims()[3]

def num_channels(layer):
    return layer.dims()[2]

def img_width(img):
    return img.dims()[0]

def img_height(img):
    return img.dims()[1]

def num_input(img):
    if len(img.dims()) < 4:
        return 1
    return img.dims()[3]

