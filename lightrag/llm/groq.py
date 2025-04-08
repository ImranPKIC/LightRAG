<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
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
=======
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
import os
from groq import AsyncGroq
from typing import Optional, Union, List, Any, AsyncGenerator
from groq import Groq
import asyncio
import json
from ..utils import logger
from ..prompt import (
    PROMPTS,
)

async def groq_stream_complete(
    prompt: str,
    model: str = "llama-3.1-8b-instant",
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.7,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: Optional[Union[str, List[str]]] = None,
    keyword_extraction: bool = False,
    hashing_kv: Any = None,
    **kwargs: Any
) -> str:
    """
    Complete text using Groq API with streaming support.
    
    Args:
        prompt: The user prompt
        model: Model name (default: llama-3.1-8b-instant)
        system_prompt: Optional system prompt
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 1024)
        top_p: Top-p sampling parameter (default: 0.7)
        frequency_penalty: Frequency penalty (default: 0.0)
        presence_penalty: Presence penalty (default: 0.0)
        stop: Optional stop sequences
        keyword_extraction: Whether this is for keyword extraction
        hashing_kv: Whether this is for hashing key-value pairs
        **kwargs: Additional arguments
        
    Returns:
        Complete generated text
    """
    try:
        # Get API key from environment or kwargs
        api_key = kwargs.pop("api_key", os.getenv("GROQ_API_KEY"))
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        client = Groq(api_key=api_key)
        
        messages = []
        # If keyword extraction, use the exact prompt from PROMPTS
        if keyword_extraction:
            # Use the exact entity extraction prompt from LightRAG
            entity_types = kwargs.get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"])
            language = kwargs.get("language", PROMPTS["DEFAULT_LANGUAGE"])
            examples = kwargs.get("examples", "")
            
            context_base = {
                "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
                "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
                "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
                "entity_types": ",".join(entity_types),
                "examples": examples,
                "language": language,
                "input_text": prompt
            }
            
            # Use the exact prompt format from LightRAG
            system_prompt = PROMPTS["entity_extraction"].format(**context_base)
            prompt = ""  # Move content to system prompt for better context
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if prompt:  # Only add user message if there's content
            messages.append({"role": "user", "content": prompt})

        # Log the request for debugging
        logger.debug(f"Sending request to Groq API with model {model}")
        logger.debug(f"Messages: {json.dumps(messages, indent=2)}")

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=True,
            stop=stop,
        )

        result = []
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                result.append(content)
                
        response = "".join(result).strip()
        logger.debug(f"Raw response from Groq: {response}")

        # If keyword extraction, ensure the response ends with completion delimiter
        if keyword_extraction:
            # Ensure response is not empty
            if not response:
                logger.warning("Empty response from model")
                return PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
                
            # Ensure response has proper format
            if not response.endswith(PROMPTS["DEFAULT_COMPLETION_DELIMITER"]):
                response += PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
                
            # Split response into records and validate
            valid_records = []
            invalid_records = []
            
            for part in response.split(PROMPTS["DEFAULT_RECORD_DELIMITER"]):
                part = part.strip()
                if not part:
                    continue
                    
                # Remove completion delimiter if present
                if part.endswith(PROMPTS["DEFAULT_COMPLETION_DELIMITER"]):
                    part = part[:-len(PROMPTS["DEFAULT_COMPLETION_DELIMITER"])].strip()
                    
                # Check if record has proper format
                if part.startswith('("') and part.endswith('")'):
                    # Validate record structure
                    try:
                        record_content = part[2:-2]  # Remove outer ("...")
                        attributes = record_content.split(PROMPTS["DEFAULT_TUPLE_DELIMITER"])
                        record_type = attributes[0].strip('"')
                        
                        # Validate record type and minimum fields
                        if record_type == "entity" and len(attributes) >= 4:
                            valid_records.append(part)
                        elif record_type == "relationship" and len(attributes) >= 5:
                            valid_records.append(part)
                        elif record_type == "content_keywords" and len(attributes) >= 2:
                            valid_records.append(part)
                        else:
                            invalid_records.append((part, f"Invalid number of fields for type {record_type}"))
                    except Exception as e:
                        invalid_records.append((part, str(e)))
                else:
                    invalid_records.append((part, "Missing proper record format"))
            
            # Log validation results
            if invalid_records:
                logger.warning(f"Found {len(invalid_records)} invalid records:")
                for record, reason in invalid_records:
                    logger.warning(f"  Invalid record: {record}")
                    logger.warning(f"  Reason: {reason}")
            
            if not valid_records:
                logger.warning("No valid records found in response")
                return PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
                
            # Reconstruct response with valid records
            response = PROMPTS["DEFAULT_RECORD_DELIMITER"].join(valid_records) + PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
            logger.debug(f"Final formatted response: {response}")

        return response

    except Exception as e:
        logger.error(f"Error in Groq completion: {str(e)}")
        logger.error(f"Full error details: {str(e.__class__.__name__)}: {str(e)}")
        if keyword_extraction:
            # Return valid empty response for keyword extraction
            return PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        raise
>>>>>>> Stashed changes
=======
        raise
>>>>>>> Stashed changes
=======
        raise
>>>>>>> Stashed changes
=======
        raise
>>>>>>> Stashed changes
=======
        raise
>>>>>>> Stashed changes
