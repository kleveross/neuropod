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

import os
import json
import shutil

from neuropod.utils.packaging_utils import packager


@packager(platform="caffe2")
def create_caffe2_neuropod(neuropod_path, input_spec, output_spec,
                           node_name_mapping, predict_net, init_net, **kwargs):
    """
    Packages a TorchScript model as a neuropod package.

    {common_doc_pre}

    :param  node_name_mapping:  Mapping from a neuropod input/output name to a blob in the net. 

                                !!! note ""
                                    ***Example***:
                                    ```
                                    {
                                        "x": "inputA",
                                        "y": "inputB",
                                        "out": "output",
                                    }
                                    ```

    :param  init_net:           path a protobuf that has all of the network weights. 

    :param  predict_net:        path a protobuf that defines the network.
    {common_doc_post}
    """

    # Create a folder to store the model
    neuropod_data_path = os.path.join(neuropod_path, "0", "data")
    os.makedirs(neuropod_data_path)

    # Add the model to the neuropod
    init_path = os.path.join(neuropod_data_path, "init_net.pb")
    predict_path = os.path.join(neuropod_data_path, "predict_net.pb")

    shutil.copyfile(init_net, init_path)
    shutil.copyfile(predict_net, predict_path)

    # Make sure we have mappings for everything in the spec
    expected_keys = set()
    for spec in [input_spec, output_spec]:
        for tensor in spec:
            expected_keys.add(tensor["name"])

    actual_keys = set(node_name_mapping.keys())
    missing_keys = expected_keys - actual_keys

    if len(missing_keys) > 0:
        raise ValueError(
            "Expected an item in `node_name_mapping` for every tensor in input_spec and output_spec. Missing: `{}`"
            .format(missing_keys))

    # We also need to save the node name mapping so we know how to run the model
    # This is tensorflow specific config so it's not saved in the overall neuropod config
    with open(os.path.join(neuropod_path, "0", "config.json"),
              "w") as config_file:
        json.dump(
            {
                "node_name_mapping": node_name_mapping,
            },
            config_file,
        )
