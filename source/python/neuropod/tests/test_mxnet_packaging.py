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
import mxnet as mx
from mxnet.gluon import nn

from neuropod.packagers import create_mxnet_neuropod
from neuropod.tests.utils import (
    get_addition_model_spec,
    check_addition_model,
)
from neuropod.utils.eval_utils import RUN_NATIVE_TESTS


class AdditionModel(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(AdditionModel, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, y):
        out = x + y
        return out


@unittest.skipIf(RUN_NATIVE_TESTS,
                 "MXNet are not supported by the native bindings")
class TestMXNetPackaging(unittest.TestCase):
    def package_simple_addition_model(self, do_fail=False):
        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")

            net = AdditionModel()
            net.initialize()
            net.hybridize()
            net.forward(mx.nd.array([1, 2, 3]), mx.nd.array([1, 2, 3]))

            create_mxnet_neuropod(
                neuropod_path=neuropod_path,
                model_name="addition_model",
                module=net,
                input_names=['data0', 'data1'],
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
