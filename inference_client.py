import grpc
from utils.label import add_meta_data
from protos import infer_pb2, infer_pb2_grpc
from google.protobuf.json_format import MessageToJson, MessageToDict


def run_test():
    with open('peoples.jpg', 'rb') as f:
        raw_image = f.read()
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = infer_pb2_grpc.InferenceStub(channel)
        im = infer_pb2.Image(raw_data=raw_image, image_id='1111')
        add_meta_data(im)
        print(MessageToJson(stub.Predict(im)))


if __name__ == '__main__':
    run_test()