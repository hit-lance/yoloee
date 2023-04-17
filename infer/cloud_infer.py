import numpy as np
import requests

session = requests.Session()


def cloud_infer(x, s=0, x_max=np.float32(0), x_min=np.float32(0)):
    model_input = {
        "model_input": x.tobytes(),
        "split_point": s.to_bytes(1, byteorder='big'),
        "x_max": x_max.tobytes(),
        "x_min": x_min.tobytes()
    }

    res = session.post("http://localhost:8080/predictions/yoloee",
                       data=model_input)

    prediction = np.frombuffer(res.content,
                               dtype=np.float32).reshape(125, 13, 13)
    return prediction


if __name__ == "__main__":
    # x = np.random.rand(855163)
    x = np.random.rand(1, 3, 416, 416).astype(np.float32)
    s = 0

    output = cloud_infer(x, s)
    # print(output)
