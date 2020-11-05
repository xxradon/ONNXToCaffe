### Code mainly come from [https://github.com/MTlab/onnx2caffe]("https://github.com/MTlab/onnx2caffe")  [https://github.com/seanxcwang/onnx2caffe](https://github.com/seanxcwang/onnx2caffe "https://github.com/seanxcwang/onnx2caffe") and [https://github.com/205418367/onnx2caffe]("https://github.com/205418367/onnx2caffe"),thanks for their contribution.
# onnx to Caffe
we can convert onnx operations to caffe layer which not only from https://github.com/BVLC/caffe but also from many other caffe modified branch like ssd-caffe,and only onnx opset_version=9 is supported.

1. Convert pytorch to Caffe by ONNX

> This tool converts [pytorch](https://github.com/pytorch/pytorch) model to Caffe model by [ONNX](https://github.com/onnx/onnx)  
only use for inference

2. Convert tensorflow to Caffe by ONNX

> you can use this repo https://github.com/onnx/tensorflow-onnx.

3. other deeplearning frame work to caffe bt ONNX

### Dependencies
* caffe (with python support)
* pytorch (optional if you  want to convert onnx)
* onnx
* onnxruntime

we recomand using protobuf 2.6.1 and install onnx from source  
```
git clone --recursive https://github.com/onnx/onnx.git
cd onnx 
python setup.py install
```
or just using pip
```bash
pip install onnx
```

### How to use
1. To convert onnx model to caffe:
```
python convertCaffe.py ./model/MobileNetV2.onnx ./model/MobileNetV2.prototxt ./model/MobileNetV2.caffemodel
```

### pytorch to onnx Tips
1. you can refer model_generator folder to learn how to generate onnx from pytorch,or just learn from [pytorch.org](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html "pytorch.org").
2. in pytorch,speeding up model with fusing batch normalization and convolution,so before convert pytorch pth model to onnx fusing  fusing batch normalization and convolution is a good choice.you may refer this 
[https://learnml.today/speeding-up-model-with-fusing-batch-normalization-and-convolution-3](https://learnml.today/speeding-up-model-with-fusing-batch-normalization-and-convolution-3 "https://learnml.today/speeding-up-model-with-fusing-batch-normalization-and-convolution-3").
3. Sometimes you need to use onnx-simplifier to simplify onnx model and then run convertCaffe.py to convert it into caffe model.

### Current support operation
* Conv
* Relu
* LeakyRelu
* PRelu
* Transpose
* ReduceMean
* MatMul
* BatchNormalization
* Add
* Mul
* Add
* Reshape
* MaxPool
* AveragePool
* GlobalAveragePool
* Dropout
* Gemm (InnerProduct only)
* Upsample ([nearest](https://github.com/jnulzl/caffe_plus "nearest") and bilinear all supported)
* Concat
* ConvTranspose
* Sigmoid
* Flatten
* Sqrt
* Softmax
* Unsqueeze
* Slice



