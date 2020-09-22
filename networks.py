#import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph import Conv2D, InstanceNorm, Sequential, Linear, SpectralNorm

#定义一些基础类
class Relu(fluid.dygraph.Layer):
    def __init__(self): 
        super(Relu, self).__init__()
    
    def forward(self, input):
        result = fluid.layers.relu(input)
        return result

class ReflectionPad2d(fluid.dygraph.Layer):
    def __init__(self, padding):
        super(ReflectionPad2d, self).__init__()
        self.padding = padding

    def forward(self, input):
        result = fluid.layers.pad2d(input,paddings=self.padding, mode="reflect")
        return result

class Spectralnorm(fluid.dygraph.Layer):
    def __init__(self,layer,dim=0,power_iter=1,eps=1e-12,dtype='float32'):
        super(Spectralnorm,self).__init__()
        self.spectral_norm = SpectralNorm(layer.weight.shape, dim, power_iter, eps, dtype)
        self.dim = dim
        self.power_iter = power_iter
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self,x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out


# 定义构建网络的类,这里是残差块
class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [ReflectionPad2d(padding=[1,1,1,1]),
                       Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim),
                       Relu()]

        conv_block += [ReflectionPad2d(padding=[1,1,1,1]),
                       Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim)]

        self.conv_block = Sequential(*conv_block)


    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = ReflectionPad2d(padding=[1,1,1,1])
        self.conv1 = Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = Relu()

        self.pad2 = ReflectionPad2d(padding=[1,1,1,1])
        self.conv2 = Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x

# 定义adaILN归一化块
class adaILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        #self.rho = fluid.layers.create_parameter(shape=[1, num_features, 1, 1],dtype='float32')
        #self.rho = fluid.layers.fill_constant(shape=[1, num_features, 1, 1],value=0.9,dtype='float32')
        self.rho = self.create_parameter([1, num_features, 1, 1],dtype='float32',default_initializer=fluid.initializer.Constant(0.9))

    def forward(self, input, gamma, beta):
        in_mean = fluid.layers.reduce_mean(input, dim=[2, 3], keep_dim=True)
        in_sub = input - in_mean
        in_var = fluid.layers.reduce_mean(fluid.layers.square(in_sub), dim=[2, 3], keep_dim=True)
        out_in = (input-in_mean) / fluid.layers.sqrt(in_var + self.eps)
        ln_mean = fluid.layers.reduce_mean(input, dim=[1, 2, 3], keep_dim=True)
        ln_sub = input - in_mean
        ln_var = fluid.layers.reduce_mean(fluid.layers.square(ln_sub), dim=[1, 2, 3], keep_dim=True)
        out_ln = (input-ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        out = fluid.layers.expand(self.rho, expand_times=[input.shape[0], 1, 1, 1]) * out_in + (1-fluid.layers.expand(self.rho,expand_times=[input.shape[0], 1, 1, 1])) * out_ln
        out = out * fluid.layers.unsqueeze(input=gamma, axes=[2,3]) + fluid.layers.unsqueeze(input=beta, axes=[2,3])
        return out

#定义ILN归一化块
class ILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = self.create_parameter(shape=[1, num_features, 1, 1],dtype='float32',default_initializer=fluid.initializer.Constant(0.0))
        self.gamma = self.create_parameter(shape=[1, num_features, 1, 1],dtype='float32',default_initializer=fluid.initializer.Constant(1.0))
        self.beta = self.create_parameter(shape=[1, num_features, 1, 1],dtype='float32',default_initializer=fluid.initializer.Constant(0.0))
        #self.rho.data.fill_(0.0)
        #self.gamma.data.fill_(1.0)
        #self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean = fluid.layers.reduce_mean(input, dim=[2, 3], keep_dim=True)
        in_sub = input - in_mean
        in_var = fluid.layers.reduce_mean(fluid.layers.square(in_sub), dim=[2, 3], keep_dim=True)
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        ln_mean = fluid.layers.reduce_mean(input, dim=[1, 2, 3], keep_dim=True)
        ln_sub = input - in_mean
        ln_var = fluid.layers.reduce_mean(fluid.layers.square(ln_sub), dim=[1, 2, 3], keep_dim=True)
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        out = fluid.layers.expand(self.rho, expand_times=[input.shape[0], 1, 1, 1]) * out_in + (1-fluid.layers.expand(self.rho,expand_times=[input.shape[0], 1, 1, 1])) * out_ln
        out = out * fluid.layers.expand(self.gamma, expand_times=[input.shape[0], 1, 1, 1]) + fluid.layers.expand(self.beta, expand_times=[input.shape[0], 1, 1, 1])

        return out

class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale = scale_factor

    def forward(self,x):
        result = fluid.layers.resize_nearest(input=x, scale=self.scale)
        return result

class LeakyRelu(fluid.dygraph.Layer):
    def __init__(self, alpha):
        super(LeakyRelu, self).__init__()
        self.alpha = alpha

    def forward(self,x):
        result = fluid.layers.leaky_relu(x, alpha=self.alpha)
        return result

class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = fluid.layers.clamp(w, min=self.clip_min, max=self.clip_max)
            module.rho.data = w

# 定义生成器
class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [ReflectionPad2d(padding=[3,3,3,3]),
                      Conv2D(num_channels=input_nc, num_filters=ngf, filter_size=7, stride=1, padding=0, bias_attr=False),
                      InstanceNorm(ngf),
                      Relu()]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [ReflectionPad2d(padding=[1,1,1,1]),
                          Conv2D(num_channels=ngf * mult, num_filters=ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=False),
                          InstanceNorm(ngf * mult * 2),
                          Relu()]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.gmp_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.conv1x1 = Conv2D(num_channels=ngf * mult * 2, num_filters=ngf * mult, filter_size=1, stride=1, act='relu', bias_attr=None)
        #self.relu = Relu()

        # Gamma, Beta block
        if self.light:
            FC = [Linear(ngf * mult, ngf * mult, act='relu', bias_attr=False),
                  Linear(ngf * mult, ngf * mult, act='relu', bias_attr=False)]
        else:
            FC = [Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, act='relu', bias_attr=False),
                  Linear(ngf * mult, ngf * mult, act='relu', bias_attr=False)]
        self.gamma = Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = Linear(ngf * mult, ngf * mult, bias_attr=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [Upsample(scale_factor=2),
                         ReflectionPad2d(padding=[1,1,1,1]),
                         Conv2D(num_channels=ngf * mult, num_filters=int(ngf * mult / 2), filter_size=3, stride=1, padding=0, bias_attr=False),
                         ILN(int(ngf * mult / 2)),
                         Relu()]

        UpBlock2 += [ReflectionPad2d(padding=[3,3,3,3]),
                     Conv2D(num_channels=ngf, num_filters=output_nc, filter_size=7, stride=1, padding=0, act='tanh', bias_attr=False)]

        self.DownBlock = Sequential(*DownBlock)
        self.FC = Sequential(*FC)
        self.UpBlock2 = Sequential(*UpBlock2)

    def forward(self, input):
        x = self.DownBlock(input)

        gap = fluid.layers.adaptive_pool2d(input=x,  pool_size=1,pool_type='avg')
        gap_logit = self.gap_fc(fluid.layers.reshape(gap, [x.shape[0], -1]))
        gap_weight = fluid.layers.transpose(list(self.gap_fc.parameters())[0], [1,0])
        gap = x * fluid.layers.unsqueeze(input=gap_weight, axes=[2,3])

        gmp = fluid.layers.adaptive_pool2d(input=x,  pool_size=1,pool_type='max')
        gmp_logit = self.gmp_fc(fluid.layers.reshape(gmp, [x.shape[0], -1]))
        gmp_weight = fluid.layers.transpose(list(self.gmp_fc.parameters())[0], [1,0])
        gmp = x * fluid.layers.unsqueeze(input=gmp_weight, axes=[2,3])

        cam_logit = fluid.layers.concat(input=[gap_logit, gmp_logit], axis=1)
        x = fluid.layers.concat(input=[gap, gmp], axis=1)
        x = self.conv1x1(x)

        heatmap = fluid.layers.reduce_sum(input=x, dim=1, keep_dim=True)

        if self.light:
            x_ = fluid.layers.adaptive_pool2d(input=x,  pool_size=1,pool_type='avg')
            x_ = self.FC(fluid.layers.reshape(x_, [x_.shape[0], -1]))
        else:
            x_ = self.FC(fluid.layers.reshape(x, [x.shape[0], -1]))
        gamma, beta = self.gamma(x_), self.beta(x_)


        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap

# 定义判别器
class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [ReflectionPad2d(padding=[1,1,1,1]),
                 Spectralnorm(Conv2D(num_channels=input_nc, num_filters=ndf, filter_size=4, stride=2, padding=0, bias_attr=None)),
                 LeakyRelu(alpha=0.2)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [ReflectionPad2d(padding=[1,1,1,1]),
                      Spectralnorm(Conv2D(num_channels=ndf * mult, num_filters=ndf * mult * 2, filter_size=4, stride=2, padding=0, bias_attr=None)),
                      #SpectralNorm(weight_shape=[12,ndf * mult *2,256 // mult,256 // mult]),
                      LeakyRelu(alpha=0.2)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [ReflectionPad2d(padding=[1,1,1,1]),
                  Spectralnorm(Conv2D(num_channels=ndf * mult, num_filters=ndf * mult * 2, filter_size=4, stride=1, padding=0, bias_attr=None)),
                  #SpectralNorm(weight_shape=[12,ndf * mult * 2,(128 // mult)-3,(128 // mult)-3]),
                  LeakyRelu(alpha=0.2)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.gmp_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.conv1x1 = Conv2D(num_channels=ndf * mult * 2, num_filters=ndf * mult, filter_size=1, stride=1, bias_attr=None)
        self.leaky_relu = LeakyRelu(alpha=0.2)

        self.pad = ReflectionPad2d(padding=[1,1,1,1])
        self.conv = Spectralnorm(Conv2D(num_channels=ndf * mult, num_filters=1, filter_size=4, stride=1, padding=0, bias_attr=False))
        self.model = Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = fluid.layers.adaptive_pool2d(input=x,  pool_size=1,pool_type='avg')
        gap_logit = self.gap_fc(fluid.layers.reshape(gap, [x.shape[0], -1]))
        gap_weight = fluid.layers.transpose(list(self.gap_fc.parameters())[0], [1,0])
        gap = x * fluid.layers.unsqueeze(input=gap_weight, axes=[2,3])

        gmp = fluid.layers.adaptive_pool2d(input=x,  pool_size=1,pool_type='max')
        gmp_logit = self.gmp_fc(fluid.layers.reshape(gmp, [x.shape[0], -1]))
        gmp_weight = fluid.layers.transpose(list(self.gmp_fc.parameters())[0], [1,0])
        gmp = x * fluid.layers.unsqueeze(input=gmp_weight, axes=[2,3])

        cam_logit = fluid.layers.concat(input=[gap_logit, gmp_logit], axis=1)
        x = fluid.layers.concat(input=[gap, gmp], axis=1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = fluid.layers.reduce_sum(input=x, dim=1, keep_dim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


#测试网络
# with fluid.dygraph.guard():
#     img = np.ones([1,3,256,256]).astype('float32')
#     img = fluid.dygraph.to_variable(img)
#     G = ResnetGenerator(input_nc=3, output_nc=3, ngf=3, n_blocks=4, 
#                         img_size=256, light=False)
#     outs,outs2,_ = G(img)
#     #生成器生成的数据形状
#     print(outs.numpy().shape,outs2.numpy().shape)

#     D1 = Discriminator(input_nc=3, ndf=3, n_layers=7)
#     D2 = Discriminator(input_nc=3, ndf=3, n_layers=5)
#     fake11,fake12,_ = D1(outs)
#     fake21,fake22,_ = D2(outs)
#     print(fake11.numpy().shape,fake12.numpy().shape,fake21.numpy().shape,fake22.numpy().shape)
