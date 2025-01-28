import torch
import re
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration

class PromptExpanderConfig:
    """Configuration settings for PromptExpander"""
    MODEL_NAME = "roborovski/superprompt-v1"
    SYSTEM_PROMPT = "Expand the following prompt to add more detail:"
    DEFAULT_SEED = 1
    SEED_MODES = ["fixed", "random", "increment"]
    DEFAULT_SEED_MODE = "fixed"
    
    # Model settings
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_REP_PENALTY = 1.2
    
class PromptExpanderNode:
    """Node for generating expanded prompts using T5 model"""
    
    def __init__(self):
        self.model_home_dir = Path.home() / ".models"
        self.model_dir = self.model_home_dir / PromptExpanderConfig.MODEL_NAME
        self.tokenizer = None
        self.model = None
        self._current_seed = PromptExpanderConfig.DEFAULT_SEED
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Enter prompt here"}),
                "seed_mode": (PromptExpanderConfig.SEED_MODES, {"default": PromptExpanderConfig.DEFAULT_SEED_MODE}),
                "seed": ("INT", {"default": PromptExpanderConfig.DEFAULT_SEED, "min": 0, "max": 0xffffffffffffffff})
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("expanded_prompt",)
    FUNCTION = "expand_prompt"
    CATEGORY = "text"

    @staticmethod
    def remove_incomplete_sentence(paragraph: str) -> str:
        """Remove incomplete sentences from the end of the text."""
        return re.sub(r'((?:\[^.!?\](?!\[.!?\]))\*+\[^.!?\\s\]\[^.!?\]\*$)', '', paragraph.rstrip())

    def _download_models(self):
        """Download the model and tokenizer files."""
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(
                PromptExpanderConfig.MODEL_NAME
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                PromptExpanderConfig.MODEL_NAME,
                torch_dtype=torch.float16
            )
            
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            self.tokenizer.save_pretrained(self.model_dir)
            self.model.save_pretrained(self.model_dir)
            
            print(f"Downloaded {PromptExpanderConfig.MODEL_NAME} model files to {self.model_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to download models: {str(e)}")

    def _load_models(self):
        """Load the model and tokenizer from local storage or download if needed."""
        if not self.model_dir.exists():
            self._download_models()
        else:
            print("Model files found. Skipping download.")

        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_dir)
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_dir,
                torch_dtype=torch.float16
            )
            print("SuperPrompt-v1 model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")

    def expand_prompt(self, prompt: str, 
                     seed_mode: str = PromptExpanderConfig.DEFAULT_SEED_MODE,
                     seed: int = PromptExpanderConfig.DEFAULT_SEED,
                     max_new_tokens: int = PromptExpanderConfig.DEFAULT_MAX_TOKENS,
                     repetition_penalty: float = PromptExpanderConfig.DEFAULT_REP_PENALTY, 
                     remove_incomplete_sentences: bool = True) -> tuple:
        """
        Generate expanded text from the input prompt.
        
        Args:
            prompt: Input text to expand
            seed_mode: Mode for seed generation ('fixed', 'random', or 'increment')
            seed: Base seed value to use
            max_new_tokens: Maximum number of new tokens to generate
            repetition_penalty: Penalty for repetition in generated text
            remove_incomplete_sentences: Whether to remove incomplete sentences
            
        Returns:
            Tuple containing the generated text
        """
        if self.tokenizer is None or self.model is None:
            self._load_models()

        # Handle seed based on mode
        if seed_mode == "fixed":
            generation_seed = seed
        elif seed_mode == "random":
            generation_seed = torch.randint(0, 0xffffffffffffffff, (1,)).item()
        else:  # increment
            generation_seed = self._current_seed
            self._current_seed += 1

        # Set up generation
        torch.manual_seed(generation_seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Prepare input
        input_text = f"{PromptExpanderConfig.SYSTEM_PROMPT}{prompt}"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        # Move model to appropriate device
        if torch.cuda.is_available():
            self.model.to('cuda')

        # Generate text
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=True
        )

        # Process output
        dirty_text = self.tokenizer.decode(outputs[0])
        text = dirty_text.replace("<pad>", "").replace("</s>", "").strip()
        
        if remove_incomplete_sentences:
            text = self.remove_incomplete_sentence(text)
        
        return (text,)

# For testing
if __name__ == "__main__":
    m = PromptExpanderNode()
    print(m.expand_prompt("a beautiful girl"))
