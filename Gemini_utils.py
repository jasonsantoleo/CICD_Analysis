import google.generativeai as genai
import json
from typing import List, Dict, Any
import os
from datetime import datetime
import time

# Configure Gemini API key (should be in environment variables)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')  # Free tier model

def detect_anomalies(records: List[Dict[str, Any]], threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Detect anomalies in CI/CD log records using Google's Gemini model.
    
    Args:
        records: List of log records (each record is a dictionary)
        threshold: Minimum anomaly probability to include in results
    
    Returns:
        List of anomalous records with added 'anomaly_prob' field
    """
    if not records:
        return []
    
    # Prepare the prompt for Gemini
    prompt = create_anomaly_detection_prompt(records)
    
    try:
        # Call Gemini API with retry logic
        response = call_gemini_with_retry(prompt, max_retries=3)
        
        if not response:
            return []
        
        # Parse the response
        anomalies = parse_anomaly_response(response.text, records)
        
        # Filter by threshold
        filtered_anomalies = [
            anomaly for anomaly in anomalies 
            if anomaly.get('anomaly_prob', 0) >= threshold
        ]
        
        return filtered_anomalies
        
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        return []

def call_gemini_with_retry(prompt: str, max_retries: int = 3) -> Any:
    """
    Call Gemini API with retry logic for rate limiting.
    
    Args:
        prompt: The prompt to send
        max_retries: Maximum number of retries
    
    Returns:
        Gemini response object or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=4000,
                temperature=0.1,  # Low temperature for consistent results
            )
            
            # Make the API call
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Check if response was blocked or empty
            if response.candidates and response.candidates[0].content.parts:
                return response
            else:
                print(f"Response was blocked or empty on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e
    
    return None

def create_anomaly_detection_prompt(records: List[Dict[str, Any]]) -> str:
    """
    Create a structured prompt for anomaly detection with Gemini.
    
    Args:
        records: List of log records
    
    Returns:
        Formatted prompt string optimized for Gemini
    """
    # Sample a subset if too many records (Gemini has generous limits but still finite)
    sample_size = min(150, len(records))  # Gemini can handle more than OpenAI
    sample_records = records[:sample_size]
    
    prompt = f"""
You are an expert DevOps engineer specializing in CI/CD pipeline analysis. Analyze the following log records for anomalies.

ANOMALY DETECTION CRITERIA:
1. Failed builds or deployments
2. Unusual execution times (significantly longer than normal)
3. Error patterns and recurring issues
4. Unexpected status codes
5. Security-related warnings or errors
6. Resource usage spikes or timeouts
7. Deployment rollbacks or reverts
8. Missing dependencies or configuration issues

INSTRUCTIONS:
- Analyze each record for potential anomalies
- Assign anomaly probability from 0.0 (normal) to 1.0 (definitely anomalous)
- Consider context and patterns across records
- Focus on actionable issues that need DevOps attention

RESPONSE FORMAT:
Respond with a JSON array containing only the anomalous records. Each record should include:
- All original fields from the input
- "anomaly_prob": float between 0.0 and 1.0
- "anomaly_reason": brief explanation of why it's flagged

Total records to analyze: {len(sample_records)}

INPUT DATA:
{json.dumps(sample_records, indent=2, default=str)}

IMPORTANT: Respond ONLY with the JSON array, no additional text or markdown formatting.

Example response format:
[
  {{
    "run_id": "abc123",
    "stage": "build",
    "status": "failed",
    "timestamp": "2024-01-15T10:30:00Z",
    "message": "Build failed due to missing dependency",
    "execution_time": 300,
    "anomaly_prob": 0.85,
    "anomaly_reason": "Build failure with dependency issue"
  }}
]
"""
    
    return prompt

def parse_anomaly_response(ai_response: str, original_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse Gemini's response and validate the data.
    
    Args:
        ai_response: Raw response from Gemini
        original_records: Original log records for reference
    
    Returns:
        List of anomalous records with probabilities
    """
    try:
        # Clean the response text
        response_text = ai_response.strip()
        
        # Remove any markdown code block markers
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        # Remove any additional text before/after JSON
        # Find the first '[' and last ']'
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            response_text = response_text[start_idx:end_idx + 1]
        
        # Parse JSON
        anomalies = json.loads(response_text.strip())
        
        # Ensure it's a list
        if not isinstance(anomalies, list):
            print("Response is not a list, attempting to wrap it")
            anomalies = [anomalies] if isinstance(anomalies, dict) else []
        
        # Validate and clean up each anomaly record
        validated_anomalies = []
        for anomaly in anomalies:
            if isinstance(anomaly, dict):
                # Ensure anomaly_prob exists and is valid
                if 'anomaly_prob' not in anomaly:
                    anomaly['anomaly_prob'] = 0.5  # Default probability
                
                prob = float(anomaly.get('anomaly_prob', 0))
                anomaly['anomaly_prob'] = max(0.0, min(1.0, prob))
                
                # Ensure anomaly_reason exists
                if 'anomaly_reason' not in anomaly:
                    anomaly['anomaly_reason'] = "Anomaly detected by AI analysis"
                
                validated_anomalies.append(anomaly)
        
        return validated_anomalies
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Raw response: {ai_response[:500]}...")  # Show first 500 chars for debugging
        
        # Fallback: try to extract any obvious anomalies from original records
        return extract_obvious_anomalies(original_records)

def extract_obvious_anomalies(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fallback function to extract obvious anomalies when AI parsing fails.
    
    Args:
        records: Original log records
    
    Returns:
        List of obviously anomalous records
    """
    anomalies = []
    
    for record in records:
        anomaly_prob = 0.0
        reasons = []
        
        # Check for failed status
        status = str(record.get('status', '')).lower()
        if status in ['failed', 'error', 'timeout', 'cancelled']:
            anomaly_prob = max(anomaly_prob, 0.8)
            reasons.append(f"Status: {status}")
        
        # Check for error keywords in message
        message = str(record.get('message', '')).lower()
        error_keywords = ['error', 'failed', 'exception', 'timeout', 'denied', 'refused']
        if any(keyword in message for keyword in error_keywords):
            anomaly_prob = max(anomaly_prob, 0.7)
            reasons.append("Error keywords in message")
        
        # Check for long execution times (if available)
        exec_time = record.get('execution_time') or record.get('duration')
        if exec_time and isinstance(exec_time, (int, float)) and exec_time > 600:  # > 10 minutes
            anomaly_prob = max(anomaly_prob, 0.6)
            reasons.append(f"Long execution time: {exec_time}s")
        
        if anomaly_prob > 0.5:
            anomaly_record = record.copy()
            anomaly_record['anomaly_prob'] = anomaly_prob
            anomaly_record['anomaly_reason'] = "; ".join(reasons)
            anomalies.append(anomaly_record)
    
    return anomalies

def describe_anomalies(anomalies: List[Dict[str, Any]]) -> str:
    """
    Generate a human-readable summary of detected anomalies using Gemini.
    
    Args:
        anomalies: List of anomalous records
    
    Returns:
        Text description of the anomalies (2-3 sentences)
    """
    if not anomalies:
        return "No significant anomalies detected in the CI/CD logs."
    
    # Prepare summary statistics
    total_anomalies = len(anomalies)
    high_prob_count = len([a for a in anomalies if a.get('anomaly_prob', 0) > 0.8])
    
    # Get patterns
    stages = [a.get('stage', 'unknown') for a in anomalies]
    statuses = [a.get('status', 'unknown') for a in anomalies]
    reasons = [a.get('anomaly_reason', 'No reason') for a in anomalies]
    
    # Count occurrences
    from collections import Counter
    stage_counts = Counter(stages)
    status_counts = Counter(statuses)
    
    most_common_stage = stage_counts.most_common(1)[0][0] if stage_counts else "unknown"
    most_common_status = status_counts.most_common(1)[0][0] if status_counts else "unknown"
    
    # Create prompt for summary
    prompt = f"""
Generate a concise 2-3 sentence professional summary for a DevOps team about these CI/CD pipeline anomalies:

STATISTICS:
- Total anomalies found: {total_anomalies}
- High-confidence anomalies (>0.8 probability): {high_prob_count}
- Most affected pipeline stage: {most_common_stage}
- Most common issue status: {most_common_status}

SAMPLE REASONS FOR ANOMALIES:
{reasons[:5]}

REQUIREMENTS:
- Write 2-3 sentences maximum
- Focus on actionable insights
- Use professional DevOps terminology
- Highlight the most critical issues
- No bullet points or lists

Provide only the summary text, no additional formatting.
"""
    
    try:
        response = call_gemini_with_retry(prompt, max_retries=2)
        
        if response and response.text:
            return response.text.strip()
        else:
            raise Exception("No response from Gemini")
        
    except Exception as e:
        print(f"Error generating summary with Gemini: {e}")
        # Fallback summary
        return f"Analysis identified {total_anomalies} anomalies in CI/CD logs, with {high_prob_count} high-confidence issues primarily affecting the {most_common_stage} stage. Most anomalies involve {most_common_status} status and require immediate DevOps attention."

# Enhanced utility functions for Gemini

def batch_analyze_large_datasets(records: List[Dict[str, Any]], batch_size: int = 100, threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Process large datasets in batches to handle Gemini's token limits efficiently.
    
    Args:
        records: All log records
        batch_size: Number of records per batch
        threshold: Anomaly probability threshold
    
    Returns:
        Combined list of all anomalies found
    """
    all_anomalies = []
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size}")
        
        batch_anomalies = detect_anomalies(batch, threshold)
        all_anomalies.extend(batch_anomalies)
        
        # Add small delay to respect rate limits
        time.sleep(1)
    
    return all_anomalies

def get_gemini_model_info():
    """
    Get information about available Gemini models and their limits.
    """
    return {
        "model": "gemini-1.5-flash",
        "max_input_tokens": 1000000,  # Very generous limit
        "max_output_tokens": 8192,
        "rate_limit": "15 RPM (requests per minute)",
        "cost": "Free tier available",
        "recommended_batch_size": 150
    }

# Configuration constants for Gemini
DEFAULT_MODEL = "gemini-1.5-flash"
PRO_MODEL = "gemini-1.5-pro"  # More capable but uses quota faster
MAX_TOKENS_PER_REQUEST = 4000
MAX_RECORDS_PER_BATCH = 150  # Gemini can handle more than OpenAI
DEFAULT_RETRY_ATTEMPTS = 3
RATE_LIMIT_DELAY = 1  # seconds between requests