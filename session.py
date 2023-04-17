import time

import numpy as np
import requests
from compress import compress
import torch

from models.device_model import DeviceModel

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Session:
    def __init__(self, split_point, threshold=1, metrics_port=8003):

        self.device_model = DeviceModel().to(device).eval()
        self.device_model.load_state_dict(
            torch.load('yoloee.pth', map_location=device))

        self.split_point = split_point
        self.threshold = threshold

        self.http_session = requests.Session()
        self.infer_url = "http://localhost:8080/predictions/yoloee"

        # Start Prometheus metric...
        # self.metrics = Metrics(metrics_port)

    @torch.no_grad()
    def device_infer(self, x):
        return self.device_model(x, self.split_point)

    def cloud_infer(self, x):
        x = x.to('cpu').numpy()

        data = {"split_point": self.split_point.to_bytes(1, byteorder='big')}

        if self.split_point != 0:
            x, x_max, x_min = compress(x)
            data["x_max"] = x_max.tobytes()
            data["x_min"] = x_min.tobytes()

        data["model_input"] = x.tobytes()

        res = self.http_session.post(url=self.infer_url, data=data)

        model_output = np.frombuffer(res.content,
                                     dtype=np.float32).reshape(125, 13, 13)
        return model_output

    def synergistic_infer(self, x):
        if self.split_point == 0:
            model_output = self.cloud_infer(x)
        elif self.split_point == 4:
            _, model_output = self.device_infer(x)
        else:
            inter, model_output = self.device_infer(x)
            model_output = self.cloud_infer(inter)

        return model_output


if __name__ == "__main__":
    session = Session(split_point=1)
    x = torch.rand(1, 3, 416, 416).to(device)

    model_output = session.synergistic_infer(x)
    print(model_output)