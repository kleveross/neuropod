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

import numpy as np
import os
import json
from six import string_types
import caffe2
from caffe2.python import workspace

from neuropod.backends.neuropod_executor import NeuropodExecutor


class Caffe2NeuropodExecutor(NeuropodExecutor):
    """
    Executes a Caffe2 neuropod
    """
    def __init__(self, neuropod_path):
        """
        Load a Caffe2 neuropod

        :param  neuropod_path:      The path to a Caffe2 neuropod package
        """
        super(Caffe2NeuropodExecutor, self).__init__(neuropod_path)

        neuropod_data_path = os.path.join(neuropod_path, "0", "data")

        # Add the model to the neuropod
        init_path = os.path.join(neuropod_data_path, "init_net.pb")
        predict_path = os.path.join(neuropod_data_path, "predict_net.pb")

        workspace.ResetWorkspace()
        with open(init_path, "rb") as f:
            init_net = f.read()
        with open(predict_path, "rb") as f:
            predict_net = f.read()
        workspace.RunNetOnce(init_net)
        workspace.CreateNet(predict_net)

        self.model = workspace.Predictor(init_net, predict_net)
        with open(os.path.join(neuropod_path, "0", "config.json"),
                  "r") as config_file:
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

        # Convert the inputs to torch tensors and move to the appropriate device

        output_dict = {}
        feed_dict = {}

        # Get the output nodes
        for node in self.neuropod_config["output_spec"]:
            neuropod_name = node["name"]

            # Get the graph node
            caffe2_name = self.node_name_mapping[neuropod_name]

            # Add it to the output name
            output_dict[neuropod_name] = caffe2_name

        # Get the input nodes
        for node in self.neuropod_config["input_spec"]:
            neuropod_name = node["name"]

            if neuropod_name not in inputs:
                continue

            # Get the graph node
            caffe2_name = self.node_name_mapping[neuropod_name]

            # Add it to the feed_dict
            feed_dict[caffe2_name] = inputs[neuropod_name]

        # forward to feed blobs
        self.model.run(feed_dict)

        neuropod_out = {}
        for k, v in output_dict.items():
            neuropod_out[k] = workspace.FetchBlob(v)
        return neuropod_out
