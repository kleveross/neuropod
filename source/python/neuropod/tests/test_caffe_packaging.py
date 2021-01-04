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

from neuropod.packagers import create_caffe_neuropod
from neuropod.tests.utils import (
    get_addition_model_spec,
    check_addition_model,
)
from neuropod.utils.eval_utils import RUN_NATIVE_TESTS

ADDITION_MODEL_SOURCE = """
name: "add"
layer {
  name: "input"
  type: "Input"
  top: "data1"
  top: "data2"
  input_param {
    shape {
      dim: 100
    }
    shape {
      dim: 100
    }
  }
}
layer{
   name:"add"
   type: "Python"
   bottom: "data1"
   bottom: "data2"
   top:"out"
   python_param {
      module: "model_ops.CustomLayer"
      layer: "AdditionLayer"
  }
}
"""

CUSTOM_OPS = """
import caffe
class AdditionLayer(caffe.Layer):
    
    def setup(self,bottom,top):
        assert len(bottom)==2,"Need 2 inputs for addition"
    
    
    def forward(self, bottom, top):
        top[0].data[...]=bottom[0].data+bottom[1].data

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)
        pass

    def backward(self, bottom, top):
        '''
        This layer does not back propagate
        '''
        pass
"""


@unittest.skipIf(RUN_NATIVE_TESTS,
                 "Caffe are not supported by the native bindings")
class TestCaffePackaging(unittest.TestCase):
    def package_simple_addition_model(self, do_fail=False):
        with TemporaryDirectory() as test_dir:
            neuropod_path = os.path.join(test_dir, "test_neuropod")
            model_code_dir = os.path.join(test_dir, "model_code")
            os.makedirs(model_code_dir)
            model_ops_dir = os.path.join(test_dir, "model_ops")
            os.makedirs(model_ops_dir)
            with open(os.path.join(model_code_dir, "model.prototxt"),
                      "w") as f:
                f.write(ADDITION_MODEL_SOURCE)

            with open(os.path.join(model_ops_dir, "CustomLayer.py"), "w") as f:
                f.write(CUSTOM_OPS)

            create_caffe_neuropod(
                neuropod_path=neuropod_path,
                model_name="addition_model",
                prototxt=os.path.join(model_code_dir, "model.prototxt"),
                node_name_mapping={
                    'x': 'data1',
                    'y': 'data2',
                    'out': 'out',
                },
                code_path_spec=[{
                    "python_root": test_dir,
                    "dirs_to_package": ["model_ops"],
                }],
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
