from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from groq import Groq
import os
from dotenv import load_dotenv


load_dotenv()


class LlmWrapper(LLM):
    api_key: Optional[str] = None
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.5
    max_tokens: int = 2048
    client: Any = None
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.5,
        max_tokens: int = 2048,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.api_key = api_key or os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("Missing API key. Set API_KEY in your environment or pass it directly.")

        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)

        # Model configuration
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        print(f"Groq Llama 3.3 (70B) initialized: {model_name}")
        print(f"Temperature={temperature}, Max tokens={max_tokens}")

    @property
    def _llm_type(self) -> str:
        # Identify this LLM for LangChain's internal registry.
        return "groq-llama-70b"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
    
        """Send prompt and return text response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )


            # Extract text and log basic token usage
            text = response.choices[0].message.content.strip()
            usage = response.usage
            print(f"Response received ({len(text)} characters)")
            print(f"Tokens used: input={usage.prompt_tokens}, output={usage.completion_tokens}")

            return text

        except Exception as e:
            print(f"Groq API call failed: {e}")
            raise


def get_llm(
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.5,
    max_tokens: int = 2048
) -> LlmWrapper:
    
    #factory function to quickly create a configured LlmWrapper instance
    return LlmWrapper(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

