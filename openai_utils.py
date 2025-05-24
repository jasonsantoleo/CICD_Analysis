# # import openai
# import json
# from typing import List, Dict, Any
# import os
# from datetime import datetime

# # Set up OpenAI API key (should be in environment variables)
# openai.api_key = os.getenv("OPENAI_API_KEY")

# def detect_anomalies(records: List[Dict[str, Any]], threshold: float = 0.5) -> List[Dict[str, Any]]:
#     """
#     Detect anomalies in CI/CD log records using OpenAI's GPT model.
    
#     Args:
#         records: List of log records (each record is a dictionary)
#         threshold: Minimum anomaly probability to include in results
    
#     Returns:
#         List of anomalous records with added 'anomaly_prob' field
#     """
#     if not records:
#         return []
    
#     # Prepare the prompt for OpenAI
#     prompt = create_anomaly_detection_prompt(records)
    
#     try:
#         # Call OpenAI API
#         response = openai.ChatCompletion.create(
#             model="gpt-4",  # or "gpt-3.5-turbo" for cheaper option
#             messages=[
#                 {
#                     "role": "system", 
#                     "content": "You are an expert DevOps engineer specializing in CI/CD pipeline analysis. Analyze log data to identify anomalies, errors, and unusual patterns."
#                 },
#                 {
#                     "role": "user", 
#                     "content": prompt
#                 }
#             ],
#             temperature=0.1,  # Low temperature for more consistent results
#             max_tokens=4000
#         )
        
#         # Parse the response
#         ai_response = response.choices[0].message.content
#         anomalies = parse_anomaly_response(ai_response, records)
        
#         # Filter by threshold
#         filtered_anomalies = [
#             anomaly for anomaly in anomalies 
#             if anomaly.get('anomaly_prob', 0) >= threshold
#         ]
        
#         return filtered_anomalies
        
#     except Exception as e:
#         print(f"Error in anomaly detection: {e}")
#         return []

# def create_anomaly_detection_prompt(records: List[Dict[str, Any]]) -> str:
#     """
#     Create a structured prompt for anomaly detection.
    
#     Args:
#         records: List of log records
    
#     Returns:
#         Formatted prompt string
#     """
#     # Sample a subset if too many records (to stay within token limits)
#     sample_size = min(100, len(records))
#     sample_records = records[:sample_size]
    
#     prompt = f"""
# Analyze the following CI/CD log records for anomalies. Look for:
# 1. Failed builds or deployments
# 2. Unusual execution times
# 3. Error patterns
# 4. Unexpected status codes
# 5. Security-related issues
# 6. Resource usage spikes
# 7. Deployment rollbacks

# For each anomalous record, provide:
# - The exact record data
# - Anomaly probability (0.0 to 1.0)
# - Brief reason for flagging

# Total records to analyze: {len(sample_records)}

# LOG DATA:
# {json.dumps(sample_records, indent=2, default=str)}

# Please respond in JSON format with an array of anomalous records, each including the original data plus 'anomaly_prob' and 'anomaly_reason' fields.

# Example response format:
# [
#   {{
#     "run_id": "abc123",
#     "stage": "build",
#     "status": "failed",
#     "timestamp": "2024-01-15T10:30:00Z",
#     "message": "Build failed due to missing dependency",
#     "anomaly_prob": 0.85,
#     "anomaly_reason": "Build failure with dependency issue"
#   }}
# ]
# """
    
#     return prompt

# def parse_anomaly_response(ai_response: str, original_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """
#     Parse OpenAI's response and match with original records.
    
#     Args:
#         ai_response: Raw response from OpenAI
#         original_records: Original log records for reference
    
#     Returns:
#         List of anomalous records with probabilities
#     """
#     try:
#         # Try to extract JSON from the response
#         # Sometimes AI wraps JSON in markdown code blocks
#         response_text = ai_response.strip()
        
#         # Remove markdown code block markers if present
#         if response_text.startswith("```json"):
#             response_text = response_text[7:]
#         if response_text.startswith("```"):
#             response_text = response_text[3:]
#         if response_text.endswith("```"):
#             response_text = response_text[:-3]
        
#         # Parse JSON
#         anomalies = json.loads(response_text.strip())
        
#         # Ensure it's a list
#         if not isinstance(anomalies, list):
#             return []
        
#         # Validate and clean up each anomaly record
#         validated_anomalies = []
#         for anomaly in anomalies:
#             if isinstance(anomaly, dict) and 'anomaly_prob' in anomaly:
#                 # Ensure anomaly_prob is a float between 0 and 1
#                 prob = float(anomaly.get('anomaly_prob', 0))
#                 anomaly['anomaly_prob'] = max(0.0, min(1.0, prob))
#                 validated_anomalies.append(anomaly)
        
#         return validated_anomalies
        
#     except (json.JSONDecodeError, ValueError) as e:
#         print(f"Error parsing AI response: {e}")
#         # Fallback: return empty list if parsing fails
#         return []

# def describe_anomalies(anomalies: List[Dict[str, Any]]) -> str:
#     """
#     Generate a human-readable summary of detected anomalies.
    
#     Args:
#         anomalies: List of anomalous records
    
#     Returns:
#         Text description of the anomalies (2-3 sentences)
#     """
#     if not anomalies:
#         return "No significant anomalies detected in the CI/CD logs."
    
#     # Prepare summary data
#     total_anomalies = len(anomalies)
#     high_prob_count = len([a for a in anomalies if a.get('anomaly_prob', 0) > 0.8])
    
#     # Get common patterns
#     stages = [a.get('stage', 'unknown') for a in anomalies]
#     statuses = [a.get('status', 'unknown') for a in anomalies]
    
#     # Count occurrences
#     stage_counts = {}
#     status_counts = {}
    
#     for stage in stages:
#         stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
#     for status in statuses:
#         status_counts[status] = status_counts.get(status, 0) + 1
    
#     # Find most common stage and status
#     most_common_stage = max(stage_counts, key=stage_counts.get) if stage_counts else "unknown"
#     most_common_status = max(status_counts, key=status_counts.get) if status_counts else "unknown"
    
#     # Create the prompt for OpenAI to generate summary
#     prompt = f"""
# Generate a concise 2-3 sentence summary of these CI/CD anomalies:

# Total anomalies: {total_anomalies}
# High probability anomalies (>0.8): {high_prob_count}
# Most affected stage: {most_common_stage}
# Most common status: {most_common_status}

# Sample anomaly reasons:
# {[a.get('anomaly_reason', 'No reason provided') for a in anomalies[:3]]}

# Provide a professional summary that a DevOps team would find useful.
# """
    
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",  # Cheaper model for simple summarization
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are a DevOps analyst. Provide concise, actionable summaries."
#                 },
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             temperature=0.3,
#             max_tokens=200
#         )
        
#         return response.choices[0].message.content.strip()
        
#     except Exception as e:
#         # Fallback summary if OpenAI call fails
#         return