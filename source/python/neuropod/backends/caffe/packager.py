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
import caffe

from neuropod.utils.packaging_utils import packager


@packager(platform="caffe")
def create_caffe_neuropod(neuropod_path,
                          input_spec,
                          output_spec,
                          node_name_mapping,
                          prototxt,
                          caffemodel=None,
                          code_path_spec=None,
                          **kwargs):
    """
    Packages a caffe model as a neuropod package.

    {common_doc_pre}

    :param  node_name_mapping:  Mapping from a neuropod input/output name to a blob in caffe.

                                !!! note ""
                                    ***Example***:
                                    ```
                                    {
                                        "x": "input0",
                                        "y": "input1",
                                        "out": "output",
                                    }
                                    ```
    :param  code_path_spec:     The folder paths of all the code that will be packaged. Note that
                                *.pyc files are ignored.

                                !!! note ""
                                    This is specified as follows:
                                    ```
                                    [{
                                        "python_root": "/some/path/to/a/python/root",
                                        "dirs_to_package": ["relative/path/to/package"]
                                    }, ...]
                                    ```

    :param  prototxt            deploy.prototxt describes the network architecture for deployment (and not training) time

    :param  caffemodel          serialized  models  as binary protocol buffer (binaryproto) files

    {common_doc_post}
    """
    # Make sure the inputs are valid

    # Create a folder to store the model
    neuropod_data_path = os.path.join(neuropod_path, "0", "data")
    os.makedirs(neuropod_data_path)

    # Copy the specified source code while preserving package paths
    if code_path_spec is not None:
        neuropod_code_path = os.path.join(neuropod_path, "0", "code")
        for copy_spec in code_path_spec:
            python_root = copy_spec["python_root"]

            for dir_to_package in copy_spec["dirs_to_package"]:
                if len(dir_to_package) == 0: continue
                shutil.copytree(
                    os.path.join(python_root, dir_to_package),
                    os.path.join(neuropod_code_path, dir_to_package),
                )

    # Add the model to the neuropod
    prototxt_path = os.path.join(neuropod_data_path, "model.prototxt")
    caffemodel_path = os.path.join(neuropod_data_path, "model.caffemodel")

    shutil.copyfile(prototxt, prototxt_path)
    if caffemodel is not None:
        shutil.copyfile(caffemodel, caffemodel_path)

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
