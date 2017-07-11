import caffe_util as util
import arrayfire as af
from collections import OrderedDict
import layers

#net_dir = '/home/patrick/af-clean/arrayfire/assets/examples/data/alexnet/'
#img_dir = '/home/patrick/af-clean/arrayfire/assets/examples/images/'
net_dir = './resources/alexnet/'
img_dir = './resources/images/'

net_prototxt = net_dir + 'deploy.prototxt'
net_data     = net_dir + 'bvlc_alexnet.caffemodel'
image_file   = img_dir + 'cat.jpg'

#This call loads caffe, sets up the network for you
caffe_net = util.load_caffe(net_prototxt, net_data)
image     = util.load_image(caffe_net, image_file)
af_net    = util.caffe_to_af(caffe_net)

#Get output from caffe's trainable layers
caffe_out = OrderedDict()
for key in caffe_net.params.viewkeys():
    caffe_out[key] = caffe_net.forward(end=key)[key].copy()
caffe_out['prob'] = caffe_net.forward()['prob'].copy()


#TODO: automate this process - read from .prototxt
#TODO: make naming intermediate layers optional
#Add layers
af_net.add_conv_layer('conv1', wx=11, wy=11, sx=4, sy=4)
af_net.add_relu_layer('relu1')
af_net.add_lrn_layer('norm1', size=5, alpha=.0001, beta=.75)
af_net.add_pool_layer('pool1', w=3, s=2)

af_net.add_conv_layer('conv2', wx=5, wy=5, sx=1, sy=1, px=2, py=2, groups=2)
af_net.add_relu_layer('relu2')
af_net.add_lrn_layer('norm2', size=5, alpha=.0001, beta=.75)
af_net.add_pool_layer('pool2', w=3, s=2)

af_net.add_conv_layer('conv3', wx=3, wy=3, px=1, py=1)
af_net.add_relu_layer('relu3')

af_net.add_conv_layer('conv4', wx=3, wy=3, px=1, py=1, groups=2)
af_net.add_relu_layer('relu4')

af_net.add_conv_layer('conv5', wx=3, wy=3, px=1, py=1, groups=2)
af_net.add_relu_layer('relu5')
af_net.add_pool_layer('pool5', w=3, s=2)

af_net.add_fc_layer('fc6')
af_net.add_relu_layer('relu6')
#dropout layer not needed for inference

af_net.add_fc_layer('fc7')
af_net.add_relu_layer('relu7')
#dropout layer not needed for inference

af_net.add_fc_layer('fc8')

af_net.add_softmax_layer('prob')

#Set input
af_net.data = image

af_out = OrderedDict()
#af_out['prob'] = af_net.forward(end = 'prob')

print "caffe"
print caffe_out['prob'].argmax()
print "arrayfire"
#print af.imax(af_out['prob'])[1]

#util.compare_output(af_out['conv4'], caffe_out['conv4'])

