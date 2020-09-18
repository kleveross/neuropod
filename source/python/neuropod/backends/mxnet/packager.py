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
import mxnet

from neuropod.utils.packaging_utils import packager


@packager(platform="mxnet")
def create_mxnet_neuropod(neuropod_path,
                          input_names=None,
                          module=None,
                          symbol_file=None,
                          param_file=None,
                          **kwargs):
    """
    Packages a MXNet model as a neuropod package.

    {common_doc_pre}

    :param  input_names         A list for model inputs names
                                !!! note ""
                                    ***Example***:
                                    ```
                                    ['data0','data1']
                                    ```

    :param  module:             An instance of a MXNet hybridize model.  If this is not provided, `symbol_file` must be set.

    :param  symbol_file:        The path to a ScriptModule that was already exported using `torch.jit.save`.
                                If this is not provided, `module` must be set.

    :param  param_file:         

    {common_doc_post}
    """
    # Make sure the inputs are valid
    if (module is None) == (symbol_file is None):
        # If they are both None or both not None
        raise ValueError(
            "Exactly one of 'module' and 'symbol_file' must be provided.")

    # Create a folder to store the model
    neuropod_data_path = os.path.join(neuropod_path, "0", "data")
    os.makedirs(neuropod_data_path)

    # Add the model to the neuropod
    symbol_path = os.path.join(neuropod_data_path, "model-symbol.json")
    param_path = os.path.join(neuropod_data_path, "model-0000.params")

    if symbol_file is not None:
        # Copy in the module
        shutil.copyfile(symbol_file, symbol_path)
        if param_file is not None:
            shutil.copyfile(param_file, param_path)
    else:
        # Save the model
        module.export(os.path.join(neuropod_data_path, 'model'))

    # save inputs name
    with open(os.path.join(neuropod_path, "0", "config.json"),
              "w") as config_file:
        json.dump(
            {"inputs": input_names},
            config_file,
        )
