## Arrayfire AlexNet Implemtation

This project is a alexnet forward pass implmentation using Arrayfire, done at the API level.

# Setup:
    edit the py_caffe variable in caffe_util.py to point to your caffe/python directory

# Run:
    python main.py

# 2 Issues:
  I'm switching the order of tuples in the code from (input, channel, x, y) to
  (x, y, channel, input) as arrayfire stores arrays colun-major. You can
  call a function with an image in the form image = (x, y, channel) but internally
  it gets switched around a lot, since I didn't know what I was doing when I started

  The convolution kernel is very slow. It's a very naive implementation, but I didn't
  really know what was going on at first. Should probably rely on the already written
  af convolve kernel if possible.