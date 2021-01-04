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
import unittest
from testpath.tempdir import TemporaryDirectory

from caffe2.python.predictor import mobile_exporter
from caffe2.python import model_helper, workspace

from neuropod.packagers import create_caffe2_neuropod
from neuropod.tests.utils import (
    get_addition_model_spec,
    check_addition_model,
)
from neuropod.utils.eval_utils import RUN_NATIVE_TESTS


def create_caffe2_addition_model():
    """
    A simple addition model
    """
    deploy_model = model_helper.ModelHelper(name="deploy_net",
                                            init_params=False)

    deploy_model.net.Add(['X', 'Y'], ['Z'])
    deploy_model.net.AddScopedExternalOutputs('Z')
    return deploy_model


def SaveNet(INIT_NET, PREDICT_NET, workspace, model):
    init_net, predict_net = mobile_exporter.Export(workspace, model.net,
                                                   model.params)

    with open(PREDICT_NET, 'wb') as f:
        f.write(model.net._net.SerializeToString())
    with open(INIT_NET, 'wb') as f:
        f.write(init_net.SerializeToString())


@unittest.skipIf(RUN_NATIVE_TESTS,
                 "ONNX are not supported by the native bindings")
class TestCaffe2Packaging(unittest.TestCase):
    def package_simple_addition_model(self, do_fail=False):
        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")

            INIT_NET = os.path.join(test_dir, "init_net.pb")
            PREDICT_NET = os.path.join(test_dir, "predict_net.pb")
            SaveNet(INIT_NET, PREDICT_NET, workspace,
                    create_caffe2_addition_model())

            create_caffe2_neuropod(
                neuropod_path=neuropod_path,
                model_name="addition_model",
                node_name_mapping={
                    "x": "X",
                    "y": "Y",
                    "out": "Z",
                },
                predict_net=PREDICT_NET,
                init_net=INIT_NET,
                # Get the input/output spec along with test data
                **get_addition_model_spec(do_fail=do_fail))

            # Run some additional checks
            check_addition_model(neuropod_path)

    def test_simple_addition_model(self):
        # Tests a case where packaging works correctly and
        # the model output matches the expected output
        self.package_simple_addition_model()

    def test_simple_addition_model_failure(self):
        # Tests a case where the output does not match the expected output
        with self.assertRaises(ValueError):
            self.package_simple_addition_model(do_fail=True)


if __name__ == "__main__":
    unittest.main()
