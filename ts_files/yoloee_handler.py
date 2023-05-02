# custom handler file

# model_handler.py
"""
ModelHandler defines a custom model handler.
"""

import math
import os
import numpy as np
import torch
import bitshuffle

from ts.torch_handler.base_handler import BaseHandler


def linear_dequantize(x_q, x_max, x_min):
    scale = (x_max - x_min) / 255
    x = x_min + scale * x_q
    return x


def uncompress(x_c, x_max, x_min, x_shape):
    x = bitshuffle.decompress_lz4(x_c, (math.prod(x_shape), ),
                                  np.dtype('uint8'))
    x = linear_dequantize(x, x_max, x_min)
    x = x.reshape(x_shape)
    return x


class YOLOEEHandler(BaseHandler):
    """
    A custom model handler implementation.
    """
    def __init__(self):
        super().__init__()
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.initialized = True

        properties = context.system_properties
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_file = self.manifest["model"].get("modelFile", "")
        serialized_file = self.manifest["model"]["serializedFile"]
        self.model_pt_path = os.path.join(model_dir, serialized_file)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = self._load_pickled_model(model_dir, model_file,
                                              self.model_pt_path)
        self.model.to(self.device)
        self.model.eval()
        #  load the model, refer 'custom handler class' above for details

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        data = data[0]

        s = int.from_bytes(data["split_point"], byteorder='big')

        if s == 0:
            model_input = np.frombuffer(data["model_input"],
                                        dtype=np.float32).reshape(
                                            1, 3, 416, 416)
        else:
            x_c = np.frombuffer(data["model_input"], dtype=np.uint8)
            x_max = np.frombuffer(data["x_max"], dtype=np.float32)
            x_min = np.frombuffer(data["x_min"], dtype=np.float32)

            if s == 1:
                x_shape = (1, 128, 52, 52) 
            elif s==2:
                x_shape = (1, 256, 26, 26)
            else:
                x_shape = (1, 512, 13, 13)

            model_input = uncompress(x_c, x_max, x_min, x_shape)

        model_input = torch.from_numpy(model_input).float()

        return model_input, s

    def inference(self, model_input, s):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        with torch.no_grad():
            model_input = model_input.to(self.device)
            model_output = self.model.forward(model_input, s)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output.to('cpu').numpy().tobytes()
        return [postprocess_output]

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        inter, s = self.preprocess(data)
        model_output = self.inference(inter, s)
        return self.postprocess(model_output)
