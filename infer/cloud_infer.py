import grpc
import numpy as np
from .inference_pb2 import PredictionsRequest
from .inference_pb2_grpc import InferenceAPIsServiceStub
# import inference_pb2_grpc

channel = grpc.insecure_channel("localhost:7070")
stub = InferenceAPIsServiceStub(channel)

model_name = "yoloee"


def cloud_infer(x, s=0, x_max=np.float32(0), x_min=np.float32(0)):
    model_input = {
        "model_input": x.tobytes(),
        "split_point": s.to_bytes(1, byteorder='big'),
        "x_max": x_max.tobytes(),
        "x_min": x_min.tobytes()
    }

    response = stub.Predictions(
        PredictionsRequest(model_name=model_name, input=model_input))

    prediction = response.prediction
    prediction = np.frombuffer(prediction,
                               dtype=np.float32).reshape(125, 13, 13)
    return prediction


if __name__ == "__main__":
    # x = np.random.rand(855163)
    x = np.random.rand(1, 1024, 26, 26)
    s = 2

    # cloud_infer(x, s, x_min, x_max)
