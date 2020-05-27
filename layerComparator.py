import caffe
import numpy as np
import onnx
import onnxruntime
from collections import OrderedDict
import os

dump_path = 'output/dump/layers'

if not os.path.exists(dump_path):
    os.makedirs(dump_path)


def getOnnxLayerOutputs(onnx_info):
    print(onnx_info)
    onnx_path = onnx_info[0]
    in_node = onnx_info[1]
    input_data = np.loadtxt(onnx_info[2])
    input_data = input_data.reshape(onnx_info[3]).astype(np.float32)

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    outputs = [x.name for x in sess.get_outputs()]
    res = sess.run(outputs, {in_node: input_data})
    res = OrderedDict(zip(outputs, res))

    output_names = list(res.keys());
    output_names.sort()
    print("onnx num of layers: {}".format(len(output_names)))

    return res


def getCaffeLayerOutputs(caffe_info):
    print(caffe_info)
    prototxt_path = caffe_info[0]
    caffemodel_path = caffe_info[1]
    in_node = caffe_info[2]
    input_data = np.loadtxt(caffe_info[3])
    input_data = input_data.reshape(caffe_info[4]).astype(np.float32)

    model = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
    model.blobs[in_node].data[...] = input_data
    model.forward()
    res = model.blobs

    output_names = list(res.keys());
    output_names.sort()
    print("caffe num of layers: {}".format(len(output_names)))

    return res


def compareLayers(onnx_info, caffe_info):
    onnx_outputs = getOnnxLayerOutputs(onnx_info)
    caffe_outputs = getCaffeLayerOutputs(caffe_info)

    for layer in onnx_outputs.keys():
        if layer in caffe_outputs.keys():
            onnx_res = onnx_outputs[layer]
            caffe_res = caffe_outputs[layer].data
            print("layer {} shape: {} for onnx vs {} for caffe"\
                   .format(layer, onnx_res.shape, caffe_res.shape))

            assert onnx_res.shape == caffe_res.shape

            dot_result = np.dot(onnx_res.flatten(), caffe_res.flatten())
            left_norm = np.sqrt(np.square(onnx_res).sum())
            right_norm = np.sqrt(np.square(caffe_res).sum())
            cos_sim = dot_result / (left_norm * right_norm)

            if cos_sim < 0.9999:
                onnx_file = os.path.join(dump_path, layer+'_onnx.txt')
                np.savetxt(onnx_file, onnx_res.flatten(), fmt='%.18f')
                caffe_file = os.path.join(dump_path, layer+'_caffe.txt')
                np.savetxt(caffe_file, caffe_res.flatten(), fmt='%.18f')
                print("cos sim of layer {}: {}".format(layer, cos_sim))

