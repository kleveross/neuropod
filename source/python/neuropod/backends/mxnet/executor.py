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
import mxnet as mx
from mxnet.gluon import nn

from neuropod.backends.neuropod_executor import NeuropodExecutor


class MxnetNeuropodExecutor(NeuropodExecutor):
    """
    Executes a MXNet neuropod
    """
    def __init__(self, neuropod_path):
        """
        Load a MXnet neuropod

        :param  neuropod_path:      The path to a MXnet neuropod package

        """
        super(MxnetNeuropodExecutor, self).__init__(neuropod_path)

        # load inputs name
        with open(os.path.join(neuropod_path, "0", "config.json"),
                  "r") as config_file:
            config = json.load(config_file)

        neuropod_data_path = os.path.join(neuropod_path, "0", "data")
        symbol_path = os.path.join(neuropod_data_path, "model-symbol.json")
        param_path = os.path.join(neuropod_data_path, "model-0000.params")

        self.model = nn.SymbolBlock.imports(symbol_file=symbol_path,
                                            input_names=config['inputs'],
                                            param_file=param_path)

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
        converted_inputs = []
        for item in self.neuropod_config["input_spec"]:
            v = inputs[item["name"]]
            converted_inputs.append(mx.ndarray.array(v, dtype=v.dtype))

        out = self.model.forward(*converted_inputs)

        if isinstance(out, mx.nd.NDArray):
            out = [
                out,
            ]

        output_spec = self.neuropod_config["output_spec"]
        neuropod_out = {
            k["name"]: v.asnumpy().astype(k["dtype"])
            for k, v in zip(output_spec, out)
        }

        return neuropod_out
