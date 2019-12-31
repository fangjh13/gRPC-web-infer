import grpc
from concurrent import futures
import logging
import numpy as np
from io import BytesIO
from mtcnn import MTCNN
import multiprocessing
from PIL import Image
from utils.label import add_meta_data
from protos.infer_pb2 import Point, Box, Landmarks, Result, InferResults
from protos import infer_pb2_grpc


class InferenceServer(infer_pb2_grpc.InferenceServicer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        process = multiprocessing.current_process()
        logger = logging.getLogger("{}-{}".format(process.name, process.pid))
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        self.logger = logger
        self.detector = MTCNN()

    def Predict(self, request, context):
        metadata = dict(context.invocation_metadata())
        self.logger.info(f"remote metada {list(metadata.items())}")
        image = Image.open(BytesIO(request.raw_data)).convert('RGB')
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


def startGrpcServer():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    server.add_insecure_port('[::]:50051')
    infer_pb2_grpc.add_InferenceServicer_to_server(InferenceServer(), server)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    for i in range(3):
        # Processes w/ SO_REUSEPORT
        # grpcio must build from source or `pip install grpcio --no-binary grpcio`
        p = multiprocessing.Process(target=startGrpcServer, args=())
        p.start()
