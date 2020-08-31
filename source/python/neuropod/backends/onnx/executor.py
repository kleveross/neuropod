# Copyright (c) 2020 UATC, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import numpy as np
import os
import six
import onnxruntime as onnxr

from neuropod.backends.neuropod_executor import NeuropodExecutor
from neuropod.utils.dtype_utils import get_dtype
from neuropod.utils.hash_utils import sha256sum

# Avoid loading the same custom op twice
loaded_op_hashes = set()


class OnnxNeuropodExecutor(NeuropodExecutor):
    """
    Executes a ONNX neuropod
    """

    def __init__(self, neuropod_path, load_custom_ops=True):
        """
        Load a ONNX neuropod

        :param  neuropod_path:  The path to a python neuropod package
        """
        super(OnnxNeuropodExecutor, self).__init__(neuropod_path)

        # Load custom ops (if any)
        if load_custom_ops and "custom_ops" in self.neuropod_config:
            #TBD :support custom ops 
            pass
        
        # Create a session
        self.sess = onnxr.InferenceSession(os.path.join(neuropod_path, "0", "data", "model.pb"))

        # Load the ONNX specific config
        with open(os.path.join(neuropod_path, "0", "config.json"), "r") as config_file:
            model_config = json.load(config_file)

            # Get the node name mapping and store it
            self.node_name_mapping = model_config["node_name_mapping"]


        

    def forward(self, inputs):
        """
        Run inference using the specifed inputs.

        :param  inputs:     A dict mapping input names to values. This must match the input
                            spec in the neuropod config for the loaded model.
                            Ex: {'x1': np.array([5]), 'x2': np.array([6])}
                            *Note:* all the keys in this dict must be strings and all the
                            values must be numpy arrays

        :returns:   A dict mapping output names to values. All the keys
                    in this dict are strings and all the values are numpy arrays.
        """

        # get the input and output nodes
        output_dict = {}
        feed_dict = {}

        # Get the output nodes
        for node in self.neuropod_config["output_spec"]:
            neuropod_name = node["name"]

            # Get the graph node
            onnx_name = self.node_name_mapping[neuropod_name]
            # Add it to the output nodes
            output_dict[neuropod_name]=onnx_name

        # Get the input nodes
        for node in self.neuropod_config["input_spec"]:
            neuropod_name = node["name"]

            # TODO(yevgeni): treat all input fields as optional at the neuropod level. If a model
            # requires a missing field it will fail therein.
            if neuropod_name not in inputs:
                continue

            # Get the graph node
            onnx_name = self.node_name_mapping[neuropod_name]

            # Add it to the feed_dict
            feed_dict[onnx_name] = inputs[neuropod_name]

        # Run inference
        outputs = self.sess.run(list(output_dict.values()), input_feed=feed_dict)
        outputs = dict(zip(output_dict.keys(),outputs))

        return outputs
