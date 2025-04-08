"""
Groq LLM Integration for LightRAG
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List
import groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

async def groq_model_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    model_name: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
    temperature: float = 0.7,
    max_tokens: int = int(os.getenv("MAX_TOKENS", "4096")),
    **kwargs: Dict[str, Any]
) -> str:
    """
    Complete text using Groq's LLM API.

    Args:
        prompt (str): The prompt text to complete
        system_prompt (Optional[str]): System prompt to guide the model's behavior
        model_name (str): Name of the Groq model to use
        temperature (float): Sampling temperature
        max_tokens (int): Maximum number of tokens to generate
        **kwargs: Additional arguments passed to the Groq client

    Returns:
        str: The generated completion text
    """
    try:
        api_key = kwargs.get('api_key') or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not found. Please provide it via kwargs or GROQ_API_KEY environment variable.")

        client = groq.Groq(api_key=api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        )

        return completion.choices[0].message.content

    except Exception as e:
        logger.error(f"Error in groq_model_complete: {str(e)}")
        raise
