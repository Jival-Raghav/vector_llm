import os
import requests
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MistralLLM:
    """
    Simple Mistral LLM interface for generating responses from retrieved context
    Takes search results and generates contextual responses
    """
    
    def __init__(self, 
                 mistral_api_key: str = None,
                 model_name: str = "mistral-small-latest"):
        """
        Initialize Mistral LLM
        
        Args:
            mistral_api_key: Mistral API key (if None, uses env variable)
            model_name: Mistral model to use
        """
        self.mistral_api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
        if not self.mistral_api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY environment variable or pass it directly.")
        
        self.model_name = model_name
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.mistral_api_key}"
        }
    
    def format_context(self, search_results: List[Dict]) -> str:
        """
        Format search results into context string
        
        Args:
            search_results: List of search results from vector database
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            # Extract source information
            source_info = f"Source: {result.get('source', 'Unknown')}"
            
            # Add additional metadata based on file type
            metadata = result.get('metadata', {})
            if result.get('file_type') == 'pdf' and 'page' in metadata:
                source_info += f" (Page {metadata['page']})"
            elif result.get('file_type') == 'csv' and 'row_number' in metadata:
                source_info += f" (Row {metadata['row_number']})"
            elif result.get('file_type') == 'json' and 'json_path' in metadata:
                source_info += f" (Path: {metadata['json_path']})"
            
            context_parts.append(f"Context {i}:\n{source_info}\nContent: {result['text']}\n")
        
        return "\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt combining query and context
        
        Args:
            query: User's question
            context: Retrieved context from vector database
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an AI assistant that answers questions based on provided context. Use the context below to answer the user's question accurately and comprehensively.

Context:
{context}

Question: {query}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information to fully answer the question, say so
- Be specific and cite relevant parts of the context when possible
- If you need to make inferences, clearly indicate that
- Keep your response focused and relevant to the question

Answer:"""
        
        return prompt
    
    def call_api(self, prompt: str, 
                 max_tokens: int = 1000,
                 temperature: float = 0.1) -> str:
        """
        Call Mistral API with the prompt
        
        Args:
            prompt: The formatted prompt
            max_tokens: Maximum tokens in response
            temperature: Creativity/randomness (0.0 to 1.0)
            
        Returns:
            LLM response text
        """
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            return f"Error calling Mistral API: {str(e)}"
        except KeyError as e:
            return f"Error parsing API response: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    
    def generate_response(self, 
                         query: str,
                         search_results: List[Dict],
                         max_tokens: int = 1000,
                         temperature: float = 0.1) -> str:
        """
        Generate response from query and search results
        
        Args:
            query: User's question
            search_results: List of search results from vector database
            max_tokens: Maximum tokens in LLM response
            temperature: LLM temperature setting
            
        Returns:
            Generated response string
        """
        if not search_results:
            return "I couldn't find any relevant information to answer your question."
        
        # Format context from search results
        context = self.format_context(search_results)
        
        # Create prompt
        prompt = self.create_prompt(query, context)
        
        # Generate response
        response = self.call_api(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response

# Convenience functions for direct usage
def generate_response_from_results(query: str, 
                                 search_results: List[Dict],
                                 mistral_api_key: str = None,
                                 max_tokens: int = 1000,
                                 temperature: float = 0.1) -> str:
    """
    Generate response directly from search results (convenience function)
    
    Args:
        query: User's question
        search_results: List of search results from vector database
        mistral_api_key: Mistral API key (optional)
        max_tokens: Maximum tokens in response
        temperature: Temperature setting
        
    Returns:
        Generated response string
    """
    llm = MistralLLM(mistral_api_key=mistral_api_key)
    return llm.generate_response(query, search_results, max_tokens, temperature)

def format_search_results(search_results: List[Dict]) -> str:
    """
    Just format search results into context (utility function)
    
    Args:
        search_results: List of search results from vector database
        
    Returns:
        Formatted context string
    """
    llm = MistralLLM()
    return llm.format_context(search_results)