import re
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from dataclasses import dataclass
from typing import Dict
import boto3

def extract_json_robust(text: str, schema: Optional[type[BaseModel]] = None) -> Dict:
    """
    Robust JSON extraction with multiple fallback strategies
    Args:
        text: Raw model output
        schema: Optional Pydantic schema for validation and defaults
    """
    
    # Step 1: Clean the text
    text = re.sub(r'<unused\d+>\w*\n?', '', text)  # Remove <unused94>thought etc
    text = re.sub(r'```json\s*', '', text)  # Remove ```json
    text = re.sub(r'```\s*', '', text)  # Remove ```
    text = text.strip()
    
    # Step 2: Try to find JSON object
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Match nested braces
        r'\{.*?\}',  # Greedy match
    ]
    
    for pattern in json_patterns:
        try:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(0)
                json_obj = json.loads(json_str)
                
                # Validate against schema if provided
                if schema:
                    try:
                        validated = schema(**json_obj)
                        if hasattr(validated, 'model_dump'):
                            return validated.model_dump()
                        elif hasattr(validated, 'dict'):
                            return validated.dict()
                        else:
                            return dict(validated)
                    except Exception as e:
                        # Schema validation failed, but we have valid JSON
                        print(f"⚠️  Schema validation failed: {e}, using raw JSON")
                        return json_obj
                
                return json_obj
        except json.JSONDecodeError:
            continue
    
    # Step 3: Try to repair common JSON errors
    try:
        # Fix incomplete JSON (missing closing brace)
        if text.count('{') > text.count('}'):
            text += '}' * (text.count('{') - text.count('}'))
        
        # Try parsing again
        json_obj = json.loads(text)
        
        if schema:
            try:
                validated = schema(**json_obj)
                if hasattr(validated, 'model_dump'):
                    return validated.model_dump()
                elif hasattr(validated, 'dict'):
                    return validated.dict()
            except:
                return json_obj
        
        return json_obj
    except:
        pass
    
    # Step 4: Extract key-value pairs manually
    try:
        result = {}
        
        # Common boolean fields
        bool_fields = ['review_excluded', 'gene_mentioned', 'cancer_relevant']
        for field in bool_fields:
            match = re.search(rf'"{field}"\s*:\s*(true|false)', text, re.IGNORECASE)
            if match:
                result[field] = match.group(1).lower() == 'true'
        
        # Common string fields
        string_fields = ['abstract_type', 'triage_evidence', 'reasoning', 'class_iii_subtype']
        for field in string_fields:
            match = re.search(rf'"{field}"\s*:\s*"([^"]*)"', text, re.DOTALL)
            if match:
                result[field] = match.group(1)
        
        # Common list fields
        list_fields = ['gene_relevance_type', 'cancer_types', 'class_i_ii_subtype']
        for field in list_fields:
            match = re.search(rf'"{field}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if match:
                items_str = match.group(1)
                items = re.findall(r'"([^"]*)"', items_str)
                result[field] = items
        
        if result and schema:
            # Fill in missing fields with defaults
            try:
                validated = schema(**result)
                if hasattr(validated, 'model_dump'):
                    return validated.model_dump()
                elif hasattr(validated, 'dict'):
                    return validated.dict()
            except:
                return result
        
        if result:
            return result
    except Exception as e:
        print(f"⚠️  Manual extraction failed: {e}")
    
    # Step 5: Return schema defaults if all else fails
    if schema:
        try:
            default_obj = schema()
            if hasattr(default_obj, 'model_dump'):
                return default_obj.model_dump()
            elif hasattr(default_obj, 'dict'):
                return default_obj.dict()
            else:
                return {
                    "raw_error": "JSON extraction failed, using defaults",
                    "content": text[:500]
                }
        except:
            pass
    
    # Complete failure
    return {
        "raw_error": "All JSON extraction strategies failed",
        "content": text[:500]
    }

# ============================================================================
# 1. DATA STRUCTURES
# ============================================================================


def call_bedrock_claude(
    client: boto3.client,
    prompt: str,
    model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
) -> Dict:
    """Call Claude or GPT via AWS Bedrock"""
    
    # Check if it's not Claude model
    not_claude = 'claude' not in model_id.lower()
    
    if not_claude:
        # GPT request format
        request_body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.0
        }
    else:
        # Claude request format
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}]
        }
    
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body)
    )
    
    response_body = json.loads(response['body'].read())
    
    # Parse usage statistics based on model type
    if not_claude:
        usage = {
            'input_tokens': response_body.get('usage', {}).get('prompt_tokens', 0),
            'output_tokens': response_body.get('usage', {}).get('completion_tokens', 0)
        }
        # Extract text response from GPT format
        text_response = response_body['choices'][0]['message']['content']
        # print("model id", model_id)
        # print("raw response:", text_response)
        # print(f"GPT Response: {text_response}")
        # only use part of the output that starts from '</reasoning>{...}' and ends at last '}'
        if "</reasoning>" in text_response:
            text_response = text_response.split("</reasoning>")[-1].strip()
            last_brace_index = text_response.rfind("}")
            if last_brace_index != -1:
                text_response = text_response[:last_brace_index + 1]
    else:
        usage = {
            'input_tokens': response_body.get('usage', {}).get('input_tokens', 0),
            'output_tokens': response_body.get('usage', {}).get('output_tokens', 0)
        }
        # Extract text response from Claude format
        text_response = response_body['content'][0]['text']
    
    # Parse JSON from response
    try:
        if "```json" in text_response:
            json_str = text_response.split("```json")[1].split("```")[0].strip()
        elif "```" in text_response:
            json_str = text_response.split("```")[1].split("```")[0].strip()
        else:
            json_str = text_response.strip()

        # Fix Python-style booleans to JSON-style (True/False -> true/false)
        json_str = json_str.replace(': True,', ': true,').replace(': False,', ': false,')
        json_str = json_str.replace(': True\n', ': true\n').replace(': False\n', ': false\n')
        json_str = json_str.replace(': True}', ': true}').replace(': False}', ': false}')

        return json.loads(json_str), usage
    except Exception as e:
        print(f"JSON parsing failed: {e}")
        print(f"Raw response text: {text_response[:500]}...")  # Print first 500 chars for debugging
        # Return empty dict to let caller handle the error appropriately
        return {}, usage


def call_llm(
    client: Optional[boto3.client],
    prompt: str,
    model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
) -> Dict:
    """
    Unified LLM caller - routes to either Claude or MedGemma
    
    Args:
        client: Bedrock client (only needed if use_medgemma=False)
        prompt: The prompt text
        model_id: Model ID for Claude (ignored if use_medgemma=True)
            
    Returns:
        Tuple of (parsed_json_response, usage_dict)
    """
    return call_bedrock_claude(client, prompt, model_id)