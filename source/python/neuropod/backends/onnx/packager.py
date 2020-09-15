import os
import json
import shutil
import onnx

from neuropod.utils.packaging_utils import packager


@packager(platform="onnx")
def create_onnx_neuropod(neuropod_path,
                         input_spec,
                         output_spec,
                         node_name_mapping,
                         onnx_model=None,
                         model_path=None,
                         **kwargs):
    """
    Packages a ONNX model as a neuropod package.

    Hint: we dont support model size > 2GiB
    {common_doc_pre}

    :param  node_name_mapping:  Mapping from a neuropod input/output name to a node in the graph. 

                                !!! note ""
                                    ***Example***:
                                    ```
                                    {
                                        "x": "Add_X",
                                        "y": "Add_Y",
                                        "out": "Add_Z",
                                    } 
                                    ```

    :param  onnx_model:             Loaded in-memory ModelProto. If this is not provided, `model_path` must be set.

    :param  model_path:             The path to a ONNX serialized ModelProto.
                                    If this is not provided, `onnx_model` must be set.
                                

    {common_doc_post}
    """
    # Make sure the inputs are valid
    if (onnx_model is None) == (model_path is None):
        # If they are both None or both not None
        raise ValueError(
            "Exactly one of 'onnx_model' and 'model_path' must be provided.")

    # Create a folder to store the model
    neuropod_data_path = os.path.join(neuropod_path, "0", "data")
    os.makedirs(neuropod_data_path)

    # Add the model to the neuropod
    target_model_path = os.path.join(neuropod_data_path, "model.pb")
    if model_path is not None:
        # Copy in the module
        shutil.copyfile(model_path, target_model_path)
    else:
        # Save the model
        onnx.save(onnx_model, target_model_path)

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
