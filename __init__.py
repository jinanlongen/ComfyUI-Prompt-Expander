from .prompt_expander_node import PromptExpanderNode

NODE_CLASS_MAPPINGS = {"PromptExpanderNode": PromptExpanderNode}
NODE_DISPLAY_NAME_MAPPINGS = {"PromptExpanderNode": "PromptExpander"}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
