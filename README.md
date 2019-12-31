# 通过谷歌gRPC部署线上机器学习模型

[**gRPC**](https://grpc.io/)是谷歌开发的远程过程调用(RPC)系统，它使用HTTP/2通信，[Protocol Buffer](https://en.wikipedia.org/wiki/Protocol_Buffers)作为接口描述语言。分为服务端和客户端，跨平台不受语言限制。

本文主要在http服务中(以下代码使用的是flask)，使用[gPRC Python](https://grpc.io/docs/tutorials/basic/python/)远程调用训练好的模型返回RESTful接口，机器学习模型是一个已训练好的人脸检测模型(mtcnn)作为演示。

所有源码托管在[github]()，可按需要查看获取，下文只列出部分主要的代码提供一些思路。

## Proto定义

使用gRPC必须先使用protocol buffers定义序列化的结构包括各对象、服务等所有类型，之后通过grpcio-tools生成服务端和客户端可用的代码，使用[proto3格式](https://developers.google.com/protocol-buffers/docs/proto3)。首先定义存放图片的`Image`用于请求参数，也就是入参是一张图片

```protobuf
// request image
message Image {
    bytes raw_data = 1;
    int32 height = 2;
    int32 width = 3;
    string image_id = 4;
    MetaData _meta_data = 5;
}
```

`message Image`定义了单张图片的存放格式主要包括`raw_data`存放图片二进制，还有图片的长高和唯一id，`_meta_data`记录各种元数据具体实现可查看上面github源码[infer.proto]()

```protobuf
// each message Result
message Result {
    Box box = 1;
    Landmarks landmarks = 2;
    double confidence = 3;
}

// return results
message InferResults {
    string image_id = 1;
    MetaData _meta_data = 2;
    repeated Result results = 3;
}
```

`message Result`定义单张人脸格式每张人脸包括`bounding box`人脸框,`landmarks`5个点和置信度`confidence`，`message InferResults`定义了单张图上所有人脸和各种元数据。

```protobuf
// run inference
service Inference {
  rpc Predict (Image) returns (InferResults) {}
}
```

`service Inference`定义了一个最简单的服务，输入一张图片输出是包含所有人脸信息的`InferResults`，就像一个函数调用一样，gRPC还支持复杂的服务比如`streaming`。

protobuf的具体格式可以查看[谷歌官网](https://developers.google.com/protocol-buffers/docs/proto3)介绍

定义完`.proto`文件后就可以生成客户端和服务端可用的接口了，需要安装`grpcio-tools`包。

```shell
python3 -m grpc_tools.protoc \
        -I ./protos \
        --python_out=./protos  \
        --grpc_python_out=./protos \
        ./protos/infer.proto
```

以上命令会生成`infer_pb2.py`和`infer_pb2_grpc.py`两个文件。

- `infer_pb2.py`中包含了我们在proto文件中定义的所有以`message`开头的类型，每个都是一个python类
- `infer_pb2_grpc.py`中包含了在proto文件中以`service`开头的类型，包括服务端需要引用`...Servicer`的类重写方法，下文重写了`Predict`方法，`add_...Servicer_to_server`也是在服务端需要添加服务到`grpc.Server`，`...Stub`类是客户端需要导入的类与服务端交互。

## 加载模型启动gRPC服务

服务端主要是继承上文生成的`infer_pb2_grpc.py`中的`InferenceServicer`重写在`infer.proto`中定义的`Predict`方法，返回指定的类型也就是`InferResults`。

```python
from protos.infer_pb2 import Point, Box, Landmarks, Result, InferResults
from protos import infer_pb2_grpc

class InferenceServicer(infer_pb2_grpc.InferenceServicer):
    """ inference server """

    def __init__(self) -> None:
        self.detector = MTCNN()  # init model

    def Predict(self, request, context):
        metadata = dict(context.invocation_metadata())
        print(f"remote metada {list(metadata.items())}")
        image = Image.open(BytesIO(request.raw_data)).convert('RGB')
        print(f'receive a image size {image.width}x{image.height}')
        infer_result_list = self.detector.detect_faces(np.array(image))
        return_results = InferResults(image_id=request.image_id)
        add_meta_data(return_results)
        for r in infer_result_list:
            x1, y1 = r['box'][:2]
            x2, y2 = r['box'][2] + x1, r['box'][3] + y1
            nose = r['keypoints']['nose']
            mouth_right = r['keypoints']['mouth_right']
            right_eye = r['keypoints']['right_eye']
            left_eye = r['keypoints']['left_eye']
            mouth_left = r['keypoints']['mouth_left']
            return_results.results.append(
                Result(box=Box(up_left=Point(x=x1, y=y1),
                               lower_right=Point(x=x2, y=y2)),
                       landmarks=Landmarks(
                           left_eye=Point(x=left_eye[0], y=left_eye[1]),
                           right_eye=Point(x=right_eye[0], y=right_eye[1]),
                           nose=Point(x=nose[0], y=nose[1]),
                           mouth_left=Point(x=mouth_left[0], y=mouth_left[1]),
                           mouth_right=Point(x=mouth_right[0], y=mouth_right[1])
                ),
                    confidence=r['confidence'])
            )
        return return_results
```

`InferenceServicer`类中`__init__`方法加载模型初始化，因为本文用的mtcnn有提供[pip包](https://pypi.org/project/mtcnn/)使用tensorflow实现，就使用默认的模型，当然你也可以使用自己的权重文件。`Predict`方法主要进行推理返回proto格式的`InferResult`。

最后一步就是启动服务端监听一个端口，客户端可以连接过来。

```python
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
server.add_insecure_port('[::]:50051')
infer_pb2_grpc.add_InferenceServicer_to_server(
    InferenceServicer(), server)
server.start()
server.wait_for_termination()
```

服务端代码[`inference_server.py`]()，之后启动即可。

```shell
python3 inference_server.py
```

## 使用gRPC客户端测试

客户端的代码简单许多构建`Image`对象给`Stub`调用即可，代码如下[`inference_client`]()

```python
from protos.infer_pb2 import Point, Box, Landmarks, Result, InferResults
from protos import infer_pb2_grp

with open('peoples.jpg', 'rb') as f:
    raw_image = f.read()
with grpc.insecure_channel('localhost:50051') as channel:
    stub = infer_pb2_grpc.InferenceStub(channel)
    im = infer_pb2.Image(raw_data=raw_image, image_id='1111')
    add_meta_data(im)
    print(stub.Predict(im))
```

以上和服务端建立连接，传入`Image`，注意得到的结果也是proto格式的，可以使用`MessageToJson`和 `MessageToDict`转换成json或者dict，还有上面`Image`中没有传`width`和`height`两个属性但我们在proto中定义了，如果不传默认就是默认值如果没有指定认值那就按照不同类型指定，参看[官方文档](https://developers.google.com/protocol-buffers/docs/proto3#default)，只要不影响服务端处理使用默认值就没什么影响。

服务端启动后，执行测试

```shell
python3 inference_client.py
```

## gRPC服务使用多进程

想启动多个模型，也就是使用多进程，一开始以为把上面服务端`futures.ThreadPoolExecutor`改成`futures.ProcessPoolExecutor`就可以了，但事实没有这么简单，不信自己动手试试就知道。

google一番后找到了答案，有两种方法可以实现参考此[issue](https://github.com/grpc/grpc/issues/16001#issuecomment-433794991)。以下使用第一种即`pre-fork + SO_REUSEPORT`

```python
def startGrpcServer():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    server.add_insecure_port('[::]:50051')
    infer_pb2_grpc.add_InferenceServicer_to_server(InferenceServer(), server)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    for i in range(3):
        p = multiprocessing.Process(target=startGrpcServer, args=())
        p.start()
```

[inference_server_multiprocess.py]()启动三个进程，需要注意的是使用此方法需要编译安装grpcio,否者会报`grpc._channel._InactiveRpcError`错，之后以上面相同的方式启动即可

```shell
pip install grpcio --no-binary grpcio
```

## 运行Flask服务

使用flask创建最简单的路由`/predict`

```python
from utils.label import ServiceClient, image_preprocess

app = Flask(__name__)
app.config['predict'] = ServiceClient(
    infer_pb2_grpc, 'InferenceStub', 'localhost', 50051)


@app.route('/predict', methods=["POST"])
def batch_features():
    res = {"message": "", "results": []}
    if request.json:
        req_dict = request.get_json()

        try:
            # convert image to protobuffer
            image = image_preprocess(req_dict)
        except Exception as e:
            current_app.logger.error(f'pre handler image error: {str(e)}')
            res['message'] = str(e)
            return jsonify(res)

        # put to predict
        try:
            remote_results = app.config['predict'].Predict(image)
            res['results'] = MessageToDict(remote_results)['results']
        except Exception as e:
            current_app.logger.error(e.details)
            res['message'] = f"inference failed: {e.code()}"
        return jsonify(res)
    else:
        res['message'] = 'please post JSON format data'
        return jsonify(res)
```

上面只是在路由中使用gRPC调用，`ServiceClient`类定义了错误处理和超时处理方面调用，代码如下

```python
class ServiceClient:
    """
    gRPC client wrapper, capture errror and can init call timeout
    """

    def __init__(self, module: infer_pb2_grpc, stub: str,
                 host: str, port: int, timeout: int = 5) -> None:
        """
        :param module: module Generated by the gRPC Python protocol compiler
        :param stub: stub name
        """
        channel = grpc.insecure_channel(f'{host}:{port}')
        try:
            grpc.channel_ready_future(channel).result(timeout=10)
        except grpc.FutureTimeoutError:
            sys.exit(f'Error connecting to {host}:{port} gRPC server, exit.')
        self.stub = getattr(module, stub)(channel)
        self.timeout = timeout

    def __getattr__(self, attr):
        return partial(self._wrapped_call, self.stub, attr)

    # args[0]: stub, args[1]: function to call, args[3]: Request
    # kwargs: keyword arguments
    def _wrapped_call(self, *args, **kwargs):
        try:
            return getattr(args[0], args[1])(
                args[2], **kwargs, timeout=self.timeout
            )
        except grpc.RpcError as e:
            print('Call {0} failed with {1}'.format(
                args[1], e.code())
            )
            raise
```

以上代码在[`web_app.py`]()和[`lable.py`]()两个文件中，测试脚本在[`test_web.py`]中。

## Reference

- [medium.com](https://medium.com/@brynmathias/kafka-and-google-protobuf-a-match-made-in-python-a1bc3381da1a)

- [github.com](https://github.com/grpc/grpc/tree/master/examples/python/multiprocessing)