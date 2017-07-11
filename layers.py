import arrayfire as af
import caffe_util as util
def __pad(image, px, py):
    if px == 0 and py == 0:
        return image

    d = image.dims()
    pad_x = d[0]+2*px
    pad_y = d[1]+2*py
    return af.moddims(af.unwrap(image, pad_x, pad_y, pad_x, pad_y, px, py),
                      pad_x, pad_y, d[2])

def conv(weights, biases, image, wx, wy, sx = 1, sy = 1, px = 0, py = 0, groups = 1):
    image = __pad(image, px, py)

    batch     = util.num_input(image)
    n_filters = util.num_filters(weights)

    n_channel = util.num_channels(weights)

    w_i = image.dims()[0]
    h_i = image.dims()[1]
    w_o = (w_i - wx) / sx + 1
    h_o = (h_i - wy) / sy + 1

    tiles     = af.unwrap(image, wx, wy, sx, sy)

    weights   = af.moddims(weights, wx*wy, n_channel, n_filters)
    out       = af.constant(0, batch, n_filters, w_o, h_o)

    if groups > 1:

        out = af.constant(0, w_o, h_o, n_filters, batch)

        split_in = util.num_channels(image) / groups
        s_i = split_in
        split_out = n_filters / groups
        s_o = split_out

        for i in xrange(groups):
            weights_slice = af.moddims(weights[:,:,i*s_o:(i+1)*s_o],
                                       wx, wy, n_channel, split_out)
            biases_slice = biases[i*s_o:(i+1)*s_o]
            image_slice  = image[:,:,i*s_i:(i+1)*s_i]
            out[:,:,i*s_o:(i+1)*s_o] = conv(weights_slice,
                                        biases_slice,
                                        image_slice,
                                        wx, wy, sx, sy, 0, 0, 1)
            # out[:,i*s_o:(i+1)*s_o] = conv(weights_slice,
            #                               biases_slice,
            #                               image_slice,
            #                               wx, wy, sx, sy, 0, 0, 1)
        return out

    #TODO: Speedup this section
    for f in xrange(n_filters):
        for d in xrange(n_channel):
            tile_d = af.reorder(tiles[:,:,d],1,0)
            weight_d = weights[:,d,f]
            out[0,f] += af.moddims(af.matmul(tile_d, weight_d), 1,1, w_o, h_o)
        out[0,f] += biases[f].to_list()[0]

    return af.reorder(out, 2, 3, 1, 0)

def relu(image):
    #TODO: Remove assumption of image.dims()[3] == 1
    if len(image.dims()) is 1:
        return af.maxof(image, af.constant(0, image.dims()[0]))
    elif len(image.dims()) is 3:
        d0, d1, d2 =  image.dims()
        return af.maxof(image, af.constant(0, d0, d1, d2))
    print "error, bad val num dims"
    return

def lrn(image, size, alpha, beta):
    #TODO: Remove assumption of image.dims()[3] == 1
    d0, d1, d2 = image.dims()
    #TODO: remove reorder and moddims call
    image = af.moddims(af.reorder(image, 2, 0, 1), 1, d2, d0, d1)
    d0, d1, d2, d3 = image.dims()

    pad = (size - 1) / 2
    padded_size = d1+2*pad
    out = af.constant(0,d1, d2, d3)
    image = af.moddims(image, d1, d2, d3)
    padded = af.moddims(af.unwrap(image, padded_size, d2, padded_size, d2, pad,0), padded_size, d2, d2)
    padded = padded*padded
    for i in xrange(size):
        out += padded[i:i+d1]
    out = 1 + (float(alpha)/size) * out
    out = af.pow(out,beta)
    #TODO: remove reorder call
    return af.reorder(af.moddims(image/out, d0,d1,d2,d3), 2, 3, 1, 0)

def pool(image, w, s):
    #TODO: Remove assumption of image.dims()[3] == 1
    d0, d1, d2 = image.dims()
    #TODO: remove reorder and moddims call
    image = af.moddims(af.reorder(image, 2, 0, 1), 1, d2, d0, d1)
    d0, d1, d2, d3 = image.dims()
    h_o = (d2 - w)/s + 1
    w_o = (d3 - w)/s + 1
    tiles = af.unwrap(af.reorder(image, 2, 3, 1, 0), w, w, s, s)
    
    return af.reorder(af.reorder(af.moddims(af.max(tiles, 0), d0, h_o, w_o,
                                            d1), 0, 3, 1, 2), 2, 3, 1, 0)
def fc(weights, biases, image):
    #TODO: fix shape of fc weights
    #TODO: fix handling 1d vs 3d image
    #TODO: remove assumption of only 1 input
    s = image.dims()[0]
    if len(image.dims()) is 3:
        in0, in1, in2 = image.dims()
        image = af.reorder(image, 1, 0, 2)
        s = in0*in1*in2

    image = af.moddims(image, 1, s)
    d0, d1, d2, d3 = weights.dims()
    weights = af.moddims(weights, d2, d3)
    out = af.moddims(af.matmul(image, weights), d3) + biases

    return out

def softmax(image):
    image = af.exp(image)
    sum = af.sum(image, 0)
    return image / sum.to_list()[0]

# def alex(net, image):
#     o1 = conv(net[0][0], net[0][1], image, 11, 4, 0, 1)
#     o2 = relu(o1)
#     o3 = lrn(o2, 5, .0001, .75)
#     o4 = pool(o3, 3, 2)
#     o4 = af.reorder(o4, 2, 3, 1, 0)
#     o5 = conv(net[1][0], net[1][1], o4, 5, 1, 2, 2)
#     o6 = relu(o5)
#     o7 = lrn(o6, 5, .0001, .75)
#     o8 = pool(o7, 3, 2)
#     o8 = af.reorder(o8, 2, 3, 1, 0)
#     o9 = conv(net[2][0], net[2][1], o8, 3, 1, 1, 1)
#     o10 = relu(o9)
#     o11 = conv(net[3][0], net[3][1], o10, 3, 1, 1, 2)
#     o12 = relu(o11)
#     o13 = conv(net[4][0], net[4][1], o12, 3, 1, 1, 2)
#     o14 = pool(o13, 3, 2)
# 
#     return o1, o5, o9, o8, o11, o13

