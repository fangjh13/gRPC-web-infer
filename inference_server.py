import grpc
from concurrent import futures
from mtcnn import MTCNN
from PIL import Image
import numpy as np
from io import BytesIO
from utils.label import add_meta_data
from protos.infer_pb2 import Point, Box, Landmarks, Result, InferResults
from protos import infer_pb2_grpc


class InferenceServicer(infer_pb2_grpc.InferenceServicer):
    """ inference server """

    def __init__(self) -> None:
        self.detector = MTCNN()  # init model

    def Predict(self, request, context):
        metadata = dict(context.invocation_metadata())
        print(f"remote metadata {list(metadata.items())}")
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


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port('[::]:50051')
    infer_pb2_grpc.add_InferenceServicer_to_server(
        InferenceServicer(), server)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
