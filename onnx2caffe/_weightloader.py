from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# from caffe import params as P
import numpy as np
from ._graph import Node, Graph

def _convert_conv(net, node, graph, err):
    weight_name = node.inputs[1]
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    node_name = node.name
    W = None
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name,))
    bias_flag = False
    bias = None
    if len(node.inputs) > 2:
        bias = node.input_tensors[node.inputs[2]]
        bias_flag = True
    # net.params[node_name][0].data = W
    # if bias_flag:
    #     net.params[node_name][1].data = bias
    np.copyto(net.params[node_name][0].data,W,casting='same_kind')
    if bias_flag:
        np.copyto(net.params[node_name][1].data, bias, casting='same_kind')

def _convert_relu(net, node, graph, err):
    pass

def _convert_sigmoid(net, node, graph, err):
    pass

def _convert_BatchNorm(net, node, graph, err):
    scale = node.input_tensors[node.inputs[1]]
    bias = node.input_tensors[node.inputs[2]]
    mean = node.input_tensors[node.inputs[3]]
    var = node.input_tensors[node.inputs[4]]
    node_name = node.name
    np.copyto(net.params[node_name + '_bn'][0].data, mean, casting='same_kind')
    np.copyto(net.params[node_name + '_bn'][1].data, var, casting='same_kind')
    net.params[node_name + '_bn'][2].data[...] = 1.0
    np.copyto(net.params[node_name][0].data, scale, casting='same_kind')
    np.copyto(net.params[node_name][1].data, bias, casting='same_kind')
    # net.params[node_name+'_bn'][1].data = var
    # net.params[node_name][0].data = scale
    # net.params[node_name][1].data = bias

def _convert_leaky_relu(net, node, graph, err):
    pass

def _convert_reduce_mean(net, node, graph, err):
    pass

def _convert_permute(net, node, graph, err):
    pass

def _convert_Add(net, node, graph, err):
    pass

def _convert_Mul(net, node, graph, err):
    pass

def _convert_Unsqueeze(net, node, graph, err):
    pass

def _convert_Squeeze(net, node, graph, err):
    pass

def _convert_Reshape(net, node, graph, err):
    pass

def _convert_Flatten(net, node, graph, err):
    pass

def _convert_pool(net, node, graph, err):
    pass

def _convert_GlobalAveragepool(net, node, graph, err):
    pass

def _convert_dropout(net, node, graph, err):
    pass

def _convert_sqrt(net, node, graph, err):
    pass

def _convert_softmax(net, node, graph, err):
    pass

def _convert_prelu(net, node, graph, err):
    node_name = node.name
    weight_name = node.inputs[1]
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name, ))

    # net.params[node_name][0].data[...] = W
    np.copyto(net.params[node_name][0].data, W.flatten(), casting='same_kind')


def _convert_matmul(net, node, graph, err):
    node_name = node.name
    weight_name = node.inputs[1]
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name, ))

    b = None
    if len(node.inputs) > 2:
        b = node.input_tensors[node.inputs[2]]
    if len(W.shape) != 2 or (b is not None and len(b.shape) != 1):
        return err.unsupported_op_configuration(node, "Gemm is supported only for inner_product layer")
    if b is not None:
        if W.shape[0] != b.shape[0]:
            return err.unsupported_op_configuration(node, "Gemm is supported only for inner_product layer")
    net.params[node_name][0].data[...] = W
    net.params[node_name][1].data[...] = b  

def _convert_gemm(net, node, graph, err):
    node_name = node.name
    weight_name = node.inputs[1]
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name, ))

    broad_cast = node.attrs.get("broadcast", 1)
    trans_b = node.attrs.get("transB", 1)
    if broad_cast != 1 or trans_b != 1:
        return err.unsupported_op_configuration(node,"Gemm is supported only for inner_product layer")

    b = None
    if len(node.inputs) > 2:
        b = node.input_tensors[node.inputs[2]]
    if len(W.shape) != 2 or (b is not None and len(b.shape) != 1):
        return err.unsupported_op_configuration(node, "Gemm is supported only for inner_product layer")
    if b is not None:
        if W.shape[0] != b.shape[0]:
            return err.unsupported_op_configuration(node, "Gemm is supported only for inner_product layer")
    net.params[node_name][0].data[...] = W
    net.params[node_name][1].data[...] = b

def _convert_upsample(net, node, graph, err):
    mode = node.attrs["mode"]
    node_name = node.name
    if  str(mode,encoding="gbk") == "nearest":
        pass
        # caffe_params = net.params[node_name][0].data
        # weights = np.ones(caffe_params.shape).astype("float32")
        # np.copyto(net.params[node_name][0].data, weights, casting='same_kind')
        # net.params[node_name][0].data[]
    elif str(mode,encoding="gbk") == "linear":
        def bilinear_weight(shape):
            weight = np.zeros(np.prod(shape), dtype='float32')
            f = np.ceil(shape[3] / 2.)
            c = (2 * f - 1 - f % 2) / (2. * f)
            for i in range(np.prod(shape)):
                x = i % shape[3]
                y = (i / shape[3]) % shape[2]
                weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            return weight.reshape(shape)

        input_name = str(node.inputs[0])

        channels = graph.channel_dims[input_name]
        scales = node.input_tensors.get(node.inputs[1])
        height_scale = int(scales[2])
        width_scale = int(scales[3])
        # caffe_params = net.params[node_name][0].data
        # weights = np.ones(caffe_params.shape).astype("float32")
        weights = bilinear_weight([channels, 1, int(2 * height_scale - height_scale % 2), int(2 * width_scale - width_scale % 2)])
        np.copyto(net.params[node_name][0].data, weights, casting='same_kind')

def _convert_concat(net, node, graph, err):
    pass

def _convert_conv_transpose(net, node, graph, err):
    weight_name = node.inputs[1]
    input_name = str(node.inputs[0])
    output_name = str(node.outputs[0])
    node_name = node.name
    W = None
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name,))
    bias_flag = False
    bias = None
    if len(node.inputs) > 2:
        bias = node.input_tensors[node.inputs[2]]
        bias_flag = True
    # net.params[node_name][0].data = W
    # if bias_flag:
    #     net.params[node_name][1].data = bias
    np.copyto(net.params[node_name][0].data,W,casting='same_kind')
    if bias_flag:
        np.copyto(net.params[node_name][1].data, bias, casting='same_kind')

_ONNX_NODE_REGISTRY = {
    "Conv": _convert_conv,
    "Relu": _convert_relu,
    "LeakyRelu": _convert_leaky_relu,
    "PRelu": _convert_prelu,
    "Transpose": _convert_permute,
    "ReduceMean": _convert_reduce_mean,
    "MatMul": _convert_matmul,
    "BatchNormalization": _convert_BatchNorm,
    "Add": _convert_Add,
    "Mul": _convert_Mul,
    "Reshape": _convert_Reshape,
    "MaxPool": _convert_pool,
    "AveragePool": _convert_pool,
    "GlobalAveragePool": _convert_GlobalAveragepool,
    "Dropout": _convert_dropout,
    "Gemm": _convert_gemm,
    "Upsample": _convert_upsample,
    "Concat": _convert_concat,
    "ConvTranspose": _convert_conv_transpose,
    "Sigmoid": _convert_sigmoid,
    "Flatten": _convert_Flatten,
    "Sqrt": _convert_sqrt,
    "Softmax": _convert_softmax,
    "Unsqueeze":_convert_Unsqueeze,
    "Squeeze":_convert_Squeeze
}


