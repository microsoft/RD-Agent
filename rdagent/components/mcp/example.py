"""Example usage of Context7 MCP integration."""

import asyncio

from context7 import query_context7


async def main():
    """Main function for testing context7 functionality."""
    error_msg = """### TRACEBACK: /opt/conda/envs/kaggle/lib/python3.11/site-packages/albumentations/__init__.py:24: UserWarning: A new version of Albumentations is available: 2.0.8 (you have 1.4.20). Upgrade using: pip install -U albumentations. To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.
check_for_updates()
/opt/conda/envs/kaggle/lib/python3.11/site-packages/pydantic/main.py:230: UserWarning: blur_limit and sigma_limit minimum value can not be both equal to 0. blur_limit minimum value changed to 3.
validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
Traceback (most recent call last):
File "/workspace/RD-Agent/git_ignore_folder/RD-Agent_workspace/55141c6414284b9f8512f998b4b91043/main.py", line 452, in <module>
main()
File "/workspace/RD-Agent/git_ignore_folder/RD-Agent_workspace/55141c6414284b9f8512f998b4b91043/main.py", line 407, in main
model = ModelClass(num_classes=num_classes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/workspace/RD-Agent/git_ignore_folder/RD-Agent_workspace/55141c6414284b9f8512f998b4b91043/main.py", line 400, in <lambda>
ModelClass = lambda num_classes=num_classes: DualBranchViT(num_classes=num_classes, pretrained=True)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/workspace/RD-Agent/git_ignore_folder/RD-Agent_workspace/55141c6414284b9f8512f998b4b91043/main.py", line 221, in __init__
self.backbone = timm.create_model(model_name, pretrained=pretrained)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/timm/models/_factory.py", line 117, in create_model
model = create_fn(
^^^^^^^^^^
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/timm/models/vision_transformer.py", line 2232, in vit_base_patch16_224
model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/timm/models/vision_transformer.py", line 2132, in _create_vision_transformer
return build_model_with_cfg(
^^^^^^^^^^^^^^^^^^^^^
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/timm/models/_builder.py", line 393, in build_model_with_cfg
pretrained_cfg = resolve_pretrained_cfg(
^^^^^^^^^^^^^^^^^^^^^^^
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/timm/models/_builder.py", line 339, in resolve_pretrained_cfg
pretrained_cfg = get_pretrained_cfg(model_with_tag)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/opt/conda/envs/kaggle/lib/python3.11/site-packages/timm/models/_registry.py", line 332, in get_pretrained_cfg
raise RuntimeError(f'Invalid pretrained tag ({tag}) for {arch_name}.')
RuntimeError: Invalid pretrained tag (dinov2) for vit_base_patch16_224. \n⚠️ Special Case: If the user mentions the 'timm' library (PyTorch Image Models), you MUST return the library ID '/huggingface/pytorch-image-models' as the most relevant match, even if other candidates exist."""

    # Normal usage (verbose=False by default)
    result = await query_context7(error_msg, verbose=True)
    print("Result:", result)

    # Debug usage with verbose output
    # result = await query_context7(error_msg, verbose=True)
    # print("Debug Result:", result)


if __name__ == "__main__":
    asyncio.run(main())
