import os
import logging
import ollama
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Union

# REMEMBER: You must run `ollama serve` before using this class to ensure 
# that the Ollama server is running and capable of handling requests.
class LocalLLM:
    """
    A local LLM (Large Language Model) class for interfacing with an Ollama-based model.
    This class provides methods to:
      1. Pull models from the Ollama repository (`get_model`).
      2. List available models (`model_list`).
      3. Generate text answers to prompts (`answer`).
      4. Generate embeddings for text data (`transform` and `transform_single`).
      5. Separate a hidden "thinking" section from an answer (`separate_thinking`).
    """

    def __init__(self, model: Optional[str] = None):
        """
        Initializes the LocalLLM object.

        :param model: 
            An optional string specifying the name or tag of the model to use.
            If not provided, defaults to the value of the "Local_MODEL" environment
            variable or 'llama2-uncensored:latest'.
        """
        # Retrieve the model name from environment variables if not provided explicitly.
        self.model: str = model or os.getenv("Local_MODEL", 'llama2-uncensored:latest')

        # Set up a logger to capture runtime information, errors, and debug messages.
        self.logger: logging.Logger = logging.getLogger("local_llm")

    def get_model(self, model: str) -> "LocalLLM":
        """
        Pulls the specified model from the Ollama repository.

        :param model: The name or tag of the model to be pulled.
        :return: Returns the current LocalLLM instance to allow method chaining.
        """
        # Pull the model using Ollama. This ensures that the model is downloaded locally.
        ollama.pull(model)
        return self
    
    def model_list(self) -> pd.DataFrame:
        """
        Retrieves a list of all models managed by the Ollama server and returns them
        as a Pandas DataFrame.

        :return:
            A DataFrame containing model information, including:
            - Model Name
            - Last Modified
            - Digest
            - Size (in MB)
            - Parameter Size
            - Quantization
        """
        # The `ollama.list()` call typically returns a structure like:
        # [("models", [Model(...), Model(...), ...])]
        results = ollama.list()
        model_data = []

        # Each element in 'results' contains a label (likely "models")
        # and a list of model objects.
        for label, model_list in results:
            for m in model_list:
                # Gather relevant metadata from each model object.
                model_name = m.model
                mod_time   = m.modified_at.strftime("%Y-%m-%d %H:%M:%S")
                digest     = m.digest
                size_mb    = f"{m.size / (1024 * 1024):.2f} MB"

                # Extract parameter size and quantization level from the model details.
                param_size = m.details.parameter_size
                quant      = m.details.quantization_level

                model_data.append([
                    model_name,
                    mod_time,
                    digest,
                    size_mb,
                    param_size,
                    quant
                ])

        # Specify column names for the DataFrame.
        headers = [
            "Model Name", 
            "Last Modified", 
            "Digest", 
            "Size (MB)", 
            "Parameter Size", 
            "Quantization"
        ]

        # Construct and return the DataFrame.
        df = pd.DataFrame(model_data, columns=headers)
        return df

    def separate_thinking(self, text: str) -> Tuple[str, str]:
        """
        Given a string containing a <think>...</think> block, this function
        separates the text inside the <think>...</think> tags from the remainder
        and returns both as a tuple (thinking, answer).

        :param text: The input string that potentially has <think>...</think>.
        :return:
            A tuple (thinking, answer), where:
              - thinking: The text inside the <think></think> tags.
              - answer:   The text outside (after) the <think></think> block.
        """
        start_tag = "<think>"
        end_tag   = "</think>"

        start_index = text.find(start_tag)
        end_index   = text.find(end_tag)

        # If the tags aren't found or are in the wrong order,
        # return empty thinking and the full text as the answer.
        if start_index == -1 or end_index == -1 or end_index < start_index:
            return "", text.strip()

        # Extract the thinking text (between <think> and </think>).
        thinking = text[start_index + len(start_tag):end_index].strip()
        # Extract everything after </think> as the answer.
        answer   = text[end_index + len(end_tag):].strip()

        return thinking, answer
    
    def answer(
        self, 
        prompt: str, 
        return_thinking: bool = False
    ) -> Union[str, Tuple[str, str]]:
        """
        Generates an answer to the given prompt using the local LLM.

        :param prompt: The input prompt for the language model.
        :param return_thinking: 
            If True, returns a tuple containing the hidden "thinking" and the final text.
            If False, returns the raw model-generated text as a single string.
        :return: 
            - If return_thinking is False, returns a single string containing the entire model response.
            - If return_thinking is True, returns a tuple (thinking, answer).
            - In case of an exception, returns an empty string.
        """
        try:
            # Send a prompt to the Ollama model and retrieve the response.
            response = ollama.chat(
                self.model, 
                messages=[{"role": "user", "content": prompt}]
            )
            # Extract the generated text from the response.
            generated_text: str = response["message"]["content"]

            # Separate out any hidden thinking block if present.
            thinking, text = self.separate_thinking(generated_text)

            if return_thinking:
                return thinking, text
            return text

        except Exception as e:
            # If an error occurs, log the error and return an empty string.
            self.logger.error(f"An error occurred in answer(): {e}")
            return ""
        
    def transform_single(self, text: str) -> np.ndarray:
        """
        Generates an embedding vector for a single piece of text using the local LLM model.

        :param text: The text to be embedded.
        :return: 
            A NumPy array containing the embedding for the input text.
        """
        # Call the Ollama embedding function, passing in the model and the text.
        ans = ollama.embed(model=self.model, input=text)
        # Convert the list of embeddings into a NumPy array and return it.
        return np.array(ans.embeddings)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of strings and returns them as a 2D NumPy array.

        :param texts: A list of strings to be embedded.
        :return: 
            A 2D NumPy array where each row corresponds to the embedding
            of a single input text.
        """
        # Transform each text string individually, then stack the results into a 2D array.
        return np.vstack([self.transform_single(text) for text in texts])
