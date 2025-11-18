# src/agents/llm_agent.py
from openai import OpenAI
from typing import List, Dict, Union
from src.config.settings import Settings
import json
import re

class ChatAgent:
    """OpenAI Chatbot client."""

    def __init__(self, config: Settings):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.openai_llm_model

    def chat(self, messages: List[dict]) -> str:
        """
        Send a list of messages to the OpenAI chat model and get the response.

        Args:
            messages: List of message dicts (e.g., [{"role": "user", "content": "Hello"}])

        Returns:
            The assistant's reply as a string.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content

    def chat_single(self, prompt: str) -> str:
        """
        Send a single prompt to the OpenAI chat model and get the response.

        Args:
            prompt: The user's prompt as a string.

        Returns:
            The assistant's reply as a string.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages)

    def label_cluster(self, feedback_texts: List[str]) -> str:
        """
        Generate a descriptive label for a cluster based on sample feedback texts.
        
        Args:
            feedback_texts: Sample customer feedback from the cluster
            
        Returns:
            A concise theme/label for the cluster
        """
        prompt = f"""You are analyzing customer feedback for Vessi waterproof shoes. Below are {len(feedback_texts)} sample comments from a cluster of similar feedback.

            Generate a concise, descriptive label (2-5 words) that captures the main theme or issue.

            Sample feedback:
            {chr(10).join(f"{i+1}. {text[:200]}" for i, text in enumerate(feedback_texts))}

            Return only the label, no explanation:"""
        
        return self.chat_single(prompt).strip()

    def tag_feedback(self, feedback_text: str, categories: List[str], allow_multiple: bool = True) -> List[str]:
        """
        Tag a single piece of customer feedback with predefined categories.
        
        Args:
            feedback_text: Customer feedback text to tag
            categories: List of possible category tags
            allow_multiple: If True, can return multiple tags; if False, returns only the best single tag
            
        Returns:
            List of category strings (single item list if allow_multiple=False)
        """
        categories_list = "\n".join(f"- {cat}" for cat in categories)
        
        if allow_multiple:
            prompt = f"""You are analyzing customer feedback for Vessi waterproof shoes. 

            Customer feedback: "{feedback_text}"

            Available categories:
            {categories_list}

            Task: Identify ALL relevant categories that apply to this feedback. Be thorough - a single piece of feedback can have multiple issues.

            IMPORTANT: 
            - Look for every distinct issue or concern mentioned
            - A review mentioning "leaking AND sizing" should get BOTH tags
            - Return ALL applicable categories, not just the primary one
            - If no categories apply, return an empty array

            Return ONLY a valid JSON array of category names that match EXACTLY as written above.

            Example responses:
            ["Waterproof Leak"]
            ["Waterproof Leak", "Sizes not standard"]
            []

            Response:"""
        else:
            prompt = f"""You are analyzing customer feedback for Vessi waterproof shoes. 

            Customer feedback: "{feedback_text}"

            Available categories:
            {categories_list}

            Task: Identify the SINGLE most important category that best describes the primary issue in this feedback.

            Return ONLY the category name exactly as written above, with no additional text or explanation.

            Response:"""
        
        response = self.chat_single(prompt).strip()
        
        if allow_multiple:
            # Try multiple parsing strategies
            tags = self._parse_json_array(response, categories)
            return tags if tags else ["Uncategorized"]
        else:
            # Single tag mode - validate and return as list
            tag = self._parse_single_tag(response, categories)
            return [tag]

    def _parse_json_array(self, response: str, categories: List[str]) -> List[str]:
        """Parse LLM response as JSON array with multiple fallback strategies."""
        # Strategy 1: Direct JSON parsing
        try:
            # Remove markdown code blocks if present
            response = re.sub(r'```json\s*|\s*```', '', response)
            tags = json.loads(response)
            if isinstance(tags, list):
                valid_tags = [tag for tag in tags if tag in categories]
                return valid_tags
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract array from text
        try:
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                tags = json.loads(match.group(0))
                if isinstance(tags, list):
                    valid_tags = [tag for tag in tags if tag in categories]
                    if valid_tags:
                        return valid_tags
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Strategy 3: Find category names in response
        response_lower = response.lower()
        found_tags = [cat for cat in categories if cat.lower() in response_lower]
        if found_tags:
            return found_tags
        
        # Strategy 4: Check if response is a single category
        if response in categories:
            return [response]
        
        for cat in categories:
            if cat.lower() == response.lower().strip('"\''):
                return [cat]
        
        return []

    def _parse_single_tag(self, response: str, categories: List[str]) -> str:
        """Parse LLM response as single tag."""
        # Clean response
        response = response.strip('"\'')
        
        # Direct match
        if response in categories:
            return response
        
        # Case-insensitive match
        for cat in categories:
            if cat.lower() == response.lower():
                return cat
        
        # Partial match
        response_lower = response.lower()
        for cat in categories:
            if cat.lower() in response_lower or response_lower in cat.lower():
                return cat
        
        return "Uncategorized"

    def tag_feedback_batch(self, feedback_texts: List[str], categories: List[str], 
                          allow_multiple: bool = True) -> List[List[str]]:
        """
        Tag multiple pieces of feedback in a single API call (more efficient).
        
        Args:
            feedback_texts: List of customer feedback texts to tag
            categories: List of possible category tags
            allow_multiple: If True, can return multiple tags per feedback
            
        Returns:
            List of tag lists (each feedback gets a list of tags)
        """
        categories_list = "\n".join(f"- {cat}" for cat in categories)
        feedback_list = "\n".join(f"{i+1}. {text}" for i, text in enumerate(feedback_texts))
        
        if allow_multiple:
            prompt = f"""You are analyzing customer feedback for Vessi waterproof shoes.

                    Available categories:
                    {categories_list}

                    Customer feedback to tag:
                    {feedback_list}

                    Task: For each piece of feedback, identify ALL relevant categories. Be thorough - look for every distinct issue mentioned.

                    IMPORTANT:
                    - A review can have multiple issues (e.g., "leaking AND uncomfortable")
                    - Return ALL applicable categories for each feedback item
                    - If no categories apply to a specific feedback, use an empty array

                    Return ONLY a valid JSON object where keys are the feedback numbers (1, 2, 3, etc.) and values are arrays of matching category names.

                    Example format:
                    {{
                    "1": ["Waterproof Leak", "Sizes not standard"],
                    "2": ["Toe Area too narrow"],
                    "3": []
                    }}

                    Response:"""
        else:
            prompt = f"""You are analyzing customer feedback for Vessi waterproof shoes.

            Available categories:
            {categories_list}

            Customer feedback to tag:
            {feedback_list}

            Task: For each piece of feedback, identify the SINGLE most important category.

            Return ONLY a valid JSON object where keys are the feedback numbers (1, 2, 3, etc.) and values are the category names.

            Example format:
            {{
            "1": "Waterproof Leak",
            "2": "Toe Area too narrow",
            "3": "Uncategorized"
            }}

            Response:"""
        
        response = self.chat_single(prompt).strip()
        
        try:
            # Remove markdown code blocks if present
            response = re.sub(r'```json\s*|\s*```', '', response)
            result_dict = json.loads(response)
            results = []
            
            for i in range(len(feedback_texts)):
                key = str(i + 1)
                if key in result_dict:
                    value = result_dict[key]
                    if allow_multiple:
                        # Ensure value is a list
                        if isinstance(value, list):
                            valid_tags = [tag for tag in value if tag in categories]
                            results.append(valid_tags if valid_tags else ["Uncategorized"])
                        elif isinstance(value, str) and value in categories:
                            # Handle case where LLM returned single string instead of array
                            results.append([value])
                        else:
                            results.append(["Uncategorized"])
                    else:
                        # Single tag mode - return as list
                        if isinstance(value, str) and value in categories:
                            results.append([value])
                        elif isinstance(value, list) and value and value[0] in categories:
                            results.append([value[0]])
                        else:
                            results.append(["Uncategorized"])
                else:
                    results.append(["Uncategorized"])
            
            return results
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to individual tagging
            print(f"Batch parsing failed: {e}. Falling back to individual tagging.")
            return [self.tag_feedback(text, categories, allow_multiple) for text in feedback_texts]