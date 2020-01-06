from flask import Flask, request, jsonify, current_app
from protos import infer_pb2_grpc
from utils.label import ServiceClient, image_preprocess
from google.protobuf.json_format import MessageToDict


app = Flask(__name__)
app.config['predict'] = ServiceClient(
    infer_pb2_grpc, 'InferenceStub', 'localhost', 50051)


@app.route('/predict', methods=["POST"])
def predict():
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


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
