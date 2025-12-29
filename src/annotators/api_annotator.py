"""API-based hallucination annotation.

Uses external LLM APIs (OpenAI, Anthropic, etc.) to verify whether
generated text contains hallucinations based on the gold answer.
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
import json
import time
import re
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import TokenAnnotation, LabeledSample, HallucinationSpan, HallucinationType, get_logger
from .base import BaseAnnotator, AnnotationResult, AnnotationConfig, AnnotationMethod

logger = get_logger(__name__)


class APIAnnotationConfig(BaseModel):
    """Configuration for API-based annotation."""
    enabled: bool = Field(default=False, description="Enable API annotation")
    provider: Literal["openai", "anthropic", "local"] = Field(
        default="openai",
        description="API provider"
    )
    model: str = Field(
        default="gpt-4",
        description="Model to use for annotation"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (if not set, uses environment variable)"
    )
    max_retries: int = Field(default=3, ge=1)
    rate_limit_delay: float = Field(default=1.0, ge=0)
    batch_size: int = Field(default=10, ge=1)
    timeout: int = Field(default=60, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    
    # Prompt customization
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt for annotation"
    )


# Default prompts for hallucination detection
DEFAULT_SYSTEM_PROMPT = """You are an expert at detecting hallucinations in AI-generated text.
A hallucination is any information in the model's response that is:
1. Not supported by the provided context/source
2. Contradicts the gold/reference answer
3. Contains factually incorrect information

Your task is to identify specific spans of text that are hallucinations."""

DEFAULT_USER_PROMPT = """Given the following:

**Question/Prompt:**
{question}

**Context/Source (if available):**
{context}

**Gold/Reference Answer:**
{gold_answer}

**Model's Response:**
{model_answer}

Please identify any hallucinations in the model's response. For each hallucination found:
1. Quote the exact text span that is hallucinated
2. Explain why it's a hallucination
3. Classify the type: "evident_conflict" (clearly wrong), "evident_baseless" (not in source), "subtle_conflict" (subtly wrong), "subtle_baseless" (subtly unsupported)

Respond in JSON format:
{{
    "has_hallucination": true/false,
    "hallucinations": [
        {{
            "text": "exact quoted text",
            "reason": "explanation",
            "type": "evident_conflict|evident_baseless|subtle_conflict|subtle_baseless"
        }}
    ],
    "overall_quality": "good|acceptable|poor",
    "confidence": 0.0-1.0
}}

If there are no hallucinations, respond with:
{{
    "has_hallucination": false,
    "hallucinations": [],
    "overall_quality": "good",
    "confidence": 1.0
}}"""


class APIAnnotator(BaseAnnotator):
    """API-based hallucination annotator.
    
    Uses external LLM APIs to verify model outputs against gold answers
    and identify hallucinated content.
    """
    
    def __init__(self, config: Optional[APIAnnotationConfig] = None):
        self.api_config = config or APIAnnotationConfig()
        super().__init__(AnnotationConfig(
            method=AnnotationMethod.API_BASED,
        ))
        self._client = None
    
    @property
    def method(self) -> AnnotationMethod:
        return AnnotationMethod.API_BASED
    
    def _get_client(self):
        """Get or create API client."""
        if self._client is not None:
            return self._client
        
        provider = self.api_config.provider
        
        if provider == "openai":
            try:
                import openai
                api_key = self.api_config.api_key
                if api_key is None:
                    import os
                    api_key = os.environ.get("OPENAI_API_KEY")
                
                if api_key is None:
                    raise ValueError("OpenAI API key not found")
                
                self._client = openai.OpenAI(api_key=api_key)
                return self._client
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        
        elif provider == "anthropic":
            try:
                import anthropic
                api_key = self.api_config.api_key
                if api_key is None:
                    import os
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
                
                if api_key is None:
                    raise ValueError("Anthropic API key not found")
                
                self._client = anthropic.Anthropic(api_key=api_key)
                return self._client
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _call_api(self, prompt: str, system_prompt: str) -> str:
        """Make API call with retries."""
        client = self._get_client()
        provider = self.api_config.provider
        
        for attempt in range(self.api_config.max_retries):
            try:
                if provider == "openai":
                    response = client.chat.completions.create(
                        model=self.api_config.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.api_config.temperature,
                        timeout=self.api_config.timeout,
                    )
                    return response.choices[0].message.content
                
                elif provider == "anthropic":
                    response = client.messages.create(
                        model=self.api_config.model,
                        max_tokens=4096,
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.api_config.temperature,
                    )
                    return response.content[0].text
                
            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < self.api_config.max_retries - 1:
                    time.sleep(self.api_config.rate_limit_delay * (attempt + 1))
                else:
                    raise
        
        raise RuntimeError("All API call attempts failed")
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from API."""
        # Try to extract JSON from response
        try:
            # First, try direct JSON parse
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Return default if parsing fails
        logger.warning(f"Failed to parse API response: {response_text[:200]}...")
        return {
            "has_hallucination": False,
            "hallucinations": [],
            "overall_quality": "unknown",
            "confidence": 0.0,
            "parse_error": True
        }
    
    def annotate(
        self,
        sample: LabeledSample,
        attention_weights: Optional[Any] = None,
        prompt_length: int = 0,
        **kwargs
    ) -> AnnotationResult:
        """Annotate a sample using API-based verification.
        
        Args:
            sample: The labeled sample
            attention_weights: Not used for API annotation
            prompt_length: Not used for API annotation
            
        Returns:
            AnnotationResult with hallucination annotations
        """
        if not self.api_config.enabled:
            logger.debug("API annotation disabled, using gold labels")
            return AnnotationResult.from_gold_labels(sample)
        
        # Prepare prompt
        system_prompt = self.api_config.system_prompt or DEFAULT_SYSTEM_PROMPT
        user_prompt = DEFAULT_USER_PROMPT.format(
            question=sample.question,
            context=sample.context or "Not provided",
            gold_answer=sample.gold_answer or "Not provided",
            model_answer=sample.answer
        )
        
        try:
            # Call API
            response_text = self._call_api(user_prompt, system_prompt)
            
            # Parse response
            parsed = self._parse_response(response_text)
            
            # Convert to AnnotationResult
            has_hallucination = parsed.get("has_hallucination", False)
            hallucinations = parsed.get("hallucinations", [])
            confidence = parsed.get("confidence", 0.5)
            
            # Create span annotations
            hallucination_spans = []
            for h in hallucinations:
                text = h.get("text", "")
                h_type = h.get("type", "unknown")
                
                # Find span in original text
                start = sample.answer.find(text)
                end = start + len(text) if start >= 0 else 0
                
                # Map type string to enum
                type_map = {
                    "evident_conflict": HallucinationType.EVIDENT_CONFLICT,
                    "evident_baseless": HallucinationType.EVIDENT_BASELESS,
                    "subtle_conflict": HallucinationType.SUBTLE_CONFLICT,
                    "subtle_baseless": HallucinationType.SUBTLE_BASELESS,
                }
                label_type = type_map.get(h_type, HallucinationType.UNKNOWN)
                
                hallucination_spans.append({
                    "text": text,
                    "start": start,
                    "end": end,
                    "label_type": label_type.value,
                    "reason": h.get("reason", ""),
                })
            
            return AnnotationResult(
                sample_id=sample.id,
                token_annotations=[],  # API provides span-level, not token-level
                has_hallucination=has_hallucination,
                hallucination_ratio=len(hallucinations) / max(1, len(sample.answer.split())),
                confidence=confidence,
                method=AnnotationMethod.API_BASED,
                metadata={
                    "provider": self.api_config.provider,
                    "model": self.api_config.model,
                    "hallucination_spans": hallucination_spans,
                    "overall_quality": parsed.get("overall_quality", "unknown"),
                    "raw_response": response_text[:500],  # Truncate for storage
                }
            )
            
        except Exception as e:
            logger.error(f"API annotation failed for {sample.id}: {e}")
            # Fall back to gold labels
            return AnnotationResult.from_gold_labels(sample)
    
    def annotate_batch(
        self,
        samples: List[LabeledSample],
        attention_weights_list: Optional[List[Any]] = None,
        prompt_lengths: Optional[List[int]] = None,
        show_progress: bool = True,
        **kwargs
    ) -> List[AnnotationResult]:
        """Annotate multiple samples with rate limiting.
        
        Args:
            samples: List of samples to annotate
            attention_weights_list: Not used
            prompt_lengths: Not used
            show_progress: Show progress bar
            
        Returns:
            List of AnnotationResults
        """
        try:
            from tqdm import tqdm
            iterator = tqdm(samples, desc="API Annotating", disable=not show_progress)
        except ImportError:
            iterator = samples
        
        results = []
        for sample in iterator:
            result = self.annotate(sample)
            results.append(result)
            
            # Rate limiting
            if self.api_config.rate_limit_delay > 0:
                time.sleep(self.api_config.rate_limit_delay)
        
        return results


def create_api_annotator(
    provider: str = "openai",
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    **kwargs
) -> APIAnnotator:
    """Factory function to create an API annotator.
    
    Args:
        provider: API provider ("openai" or "anthropic")
        model: Model to use
        api_key: API key (optional, uses env var if not provided)
        **kwargs: Additional config options
        
    Returns:
        Configured APIAnnotator
    """
    config = APIAnnotationConfig(
        enabled=True,
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs
    )
    return APIAnnotator(config)
