import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, Dict, Any

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# External DataToText Class
# ------------------------------------------------------------------------------
class DataToText(BaseEstimator, TransformerMixin):
    """
    A transformer that converts each row of a DataFrame into a natural language prompt.

    This class uses a LocalLLM instance (e.g. Ollama-based) to rephrase structured data
    into a fluent sentence. If a target-to-text mapping is provided, a final sentence is appended.
    """
    def __init__(
        self,
        data_info: Optional[str] = None,
        target_to_text_map: Optional[Dict[str, str]] = None,
        model_name: str = 'deepseek-r1:latest',
        verbose: bool = True
    ) -> None:
        self.data_info = data_info
        self.target_to_text_map = target_to_text_map
        self.model_name = model_name
        self.verbose = verbose

        # Create a LocalLLM instance (ensure the import path is correct)
        from coco_grape.data_graphicalizer.text.local_llm import LocalLLM
        self.local_llm = LocalLLM(self.model_name)
        
        # Pre-construct a final sentence if a mapping is provided.
        if self.target_to_text_map is not None:
            mapped_labels = list(self.target_to_text_map.values())
            self.final_sentence = (
                "Based on the above description, please choose the most appropriate category "
                f"from the following options: {mapped_labels}. Answer:"
            )
        else:
            self.final_sentence = ""
    
    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "DataToText":
        # No fitting is needed.
        return self
    
    def transform(self, X: pd.DataFrame) -> List[str]:
        prompts = []
        for _, row in X.iterrows():
            # Build a description with an example.
            description = (
                "The task is to rewrite information extracted from a database as a fluent sentence: \n"
                "For example, for a row with: \n"
                "Name: Alice, Age: 30, City: New York \n"
                "the corresponding fluent sentence is: \n"
                "Alice is a 30-year-old living in New York. \n\n"
                "To write a fluent single sentence, consider the following information: \n"
            )
            # Remove extra whitespace.
            description = "\n".join(line.strip() for line in description.splitlines())
            if self.data_info is not None:
                description += self.data_info + " "
            description += "Here is the instance to convert into a single sentence: \n"
            description += ", ".join([f"{col}: {row[col]}" for col in row.index])
            description += (
                "\nConvert the following structured data into a single descriptive sentence. "
                "Do not add explanations, formatting, or extra commentary. Just output the sentence itself."
            )
            # Get the fluent sentence from the LocalLLM.
            fluent_sentence = self.local_llm.answer(description)
            if self.final_sentence:
                final_prompt = f"{fluent_sentence} {self.final_sentence}"
            else:
                final_prompt = fluent_sentence
            if self.verbose:
                print('_' * 100)
                print(final_prompt)
                print('_' * 100)
            prompts.append(final_prompt)
        return prompts

# ------------------------------------------------------------------------------
# PretrainedLLMProbabilisticEstimator
# ------------------------------------------------------------------------------
class PretrainedLLMProbabilisticEstimator(BaseEstimator):
    """
    A scikit-learn style estimator that computes a probability distribution over candidate
    class tokens using a pretrained language model. It uses the external DataToText transformer
    to convert each row of a DataFrame into a natural language prompt.

    Parameters:
      - prob_llm_model_name: Name of the model used to compute token-level probabilities.
      - rephrase_llm_model_name: Name of the model used by DataToText to rephrase data rows.
      - alpha: Temperature scaling factor for probability adjustment.
      - data_info: Additional textual information appended to the prompt.
      - target_to_text_map: Mapping from target labels to textual descriptions.
      - verbose: If True, prints additional information.
    """
    def __init__(
        self,
        prob_llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        rephrase_llm_model_name: str = 'deepseek-r1:latest',
        alpha: float = 1.0,
        data_info: Optional[str] = None,
        target_to_text_map: Optional[Dict[str, str]] = None,
        verbose: bool = True
    ) -> None:
        self.prob_llm_model_name = prob_llm_model_name
        self.rephrase_llm_model_name = rephrase_llm_model_name
        self.alpha = alpha
        self.data_info = data_info
        self.target_to_text_map = target_to_text_map
        self.verbose = verbose

        # Instantiate the external DataToText transformer.
        self.data_to_text = DataToText(
            data_info=self.data_info,
            target_to_text_map=self.target_to_text_map,
            model_name=self.rephrase_llm_model_name,
            verbose=verbose
        )
        
        # Define device_map. Here we force the model onto CPU.
        device_map = {"": "cpu"}
        
        # Choose the appropriate torch dtype.
        # If the model is forced on CPU, use torch.float32.
        if device_map[""] == "cpu":
            dtype = torch.float32
        else:
            if torch.cuda.is_available():
                dtype = torch.float16
            elif torch.backends.mps.is_available():
                dtype = torch.float16
            else:
                dtype = torch.float32
        
        # Load the tokenizer and language model.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.prob_llm_model_name,
            trust_remote_code=True
        )
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.prob_llm_model_name,
            torch_dtype=dtype,
            device_map=device_map
        )
        self.llm_model.eval()
    
    def fit(self, X: Any = None, y: Any = None) -> "PretrainedLLMProbabilisticEstimator":
        # Dummy fit to satisfy the scikit-learn interface.
        return self
    
    def predict_proba(self, X: pd.DataFrame, class_labels: List[str]) -> np.ndarray:
        """
        Given a DataFrame X and a list of candidate class labels, first convert each row
        into a natural language prompt using DataToText, then compute a probability distribution
        over candidate tokens for the next token based on the language model.
        """
        prompts = self.data_to_text.transform(X)
        results = []
        for prompt in prompts:
            # Tokenize the prompt and move inputs to the model's device.
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
            
            # Determine maximum allowed positions.
            if hasattr(self.llm_model.config, "n_positions"):
                max_positions = self.llm_model.config.n_positions
            elif hasattr(self.llm_model.config, "max_position_embeddings"):
                max_positions = self.llm_model.config.max_position_embeddings
            else:
                max_positions = 1024
            
            if inputs["input_ids"].shape[1] > max_positions:
                if self.verbose:
                    print(f"Input length ({inputs['input_ids'].shape[1]}) exceeds maximum ({max_positions}). Truncating.")
                inputs["input_ids"] = inputs["input_ids"][:, -max_positions:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -max_positions:]
            
            with torch.no_grad():
                outputs = self.llm_model(**inputs)
            logits = outputs.logits  # Shape: (1, sequence_length, vocab_size)
            last_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)
            
            candidate_ids = []
            for label in class_labels:
                # Use the target-to-text mapping to adjust the candidate text if provided.
                candidate_text = self.target_to_text_map.get(label, label) if self.target_to_text_map else label
                tokens = self.tokenizer.encode(candidate_text, add_special_tokens=False)
                if len(tokens) == 0:
                    raise ValueError(f"Label '{label}' (as '{candidate_text}') could not be tokenized.")
                candidate_ids.append(tokens[0])
            candidate_logits = last_token_logits[candidate_ids]
            candidate_probs = F.softmax(candidate_logits, dim=0)
            if self.alpha != 1.0:
                candidate_probs = candidate_probs ** self.alpha
                candidate_probs = candidate_probs / candidate_probs.sum()
            results.append(candidate_probs.cpu().numpy())
        return np.array(results)

# ------------------------------------------------------------------------------
# BayesianLLMClassifier
# ------------------------------------------------------------------------------
class BayesianLLMClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that combines:
      - Likelihood estimates from an inner estimator (e.g., RandomForestClassifier),
      - A pretrained class prior computed via a language model (using PretrainedLLMProbabilisticEstimator), and
      - Class fractions computed from the training data.

    The final class probability is the normalized product of these three components.

    Note: Instead of explicitly providing booleans to indicate which estimator to use,
    these flags are automatically inferred based on whether a pretrained_llm_prob_estimator
    and/or ml_prob_estimator is provided.
    """
    def __init__(
        self,
        pretrained_llm_prob_estimator: Optional[PretrainedLLMProbabilisticEstimator] = None,
        ml_prob_estimator: Optional[Any] = None,
        target_col: Optional[str] = None,
        alpha: float = 1.0,  # Smoothing factor for ml_prob_estimator predictions.
        verbose: bool = True
    ) -> None:
        self.target_col = target_col
        self.verbose = verbose
        self.alpha = alpha

        # Store the provided estimators (if any).
        self.pretrained_llm_prob_estimator = pretrained_llm_prob_estimator
        self.ml_prob_estimator = ml_prob_estimator

        # Infer usage flags based on whether the estimators were provided.
        self.use_pretrained_llm_prob_estimator = pretrained_llm_prob_estimator is not None
        self.use_ml_prob_estimator = ml_prob_estimator is not None
    
    def _prepare_data(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare two versions of the input DataFrame:
          - X_llm: the original DataFrame (for generating LLM prompts),
          - X_inner: one-hot encoded version (using pd.get_dummies) for the inner estimator.
        """
        X_llm = X.copy()
        X_inner = pd.get_dummies(X_llm)
        return X_llm, X_inner
    
    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None) -> "BayesianLLMClassifier":
        # If target_col is provided and y is not, extract y from X.
        if self.target_col is not None and y is None:
            y = X[self.target_col]
            X = X.drop(columns=[self.target_col])
        
        self.X_llm_, X_inner = self._prepare_data(X)
        self.inner_columns_ = X_inner.columns
        
        if self.use_ml_prob_estimator:
            self.ml_prob_estimator.fit(X_inner, y)
        
        # Determine a consistent class order.
        if hasattr(y, "cat") and hasattr(y.cat, "categories"):
            self.classes_ = y.cat.categories.to_numpy()
        else:
            self.classes_ = np.unique(y)
        
        # Obtain target_to_text_map from the pretrained LLM estimator, if available.
        tt_map = self.pretrained_llm_prob_estimator.target_to_text_map if self.use_pretrained_llm_prob_estimator else None
        if tt_map is not None:
            mapped_classes = list(tt_map.keys())
            final_order = [c for c in self.classes_ if c in mapped_classes]
            remainder = [c for c in self.classes_ if c not in final_order]
            self.classes_ = np.array(final_order + remainder)
        
        # Compute class priors (fractions) in the final order.
        y_arr = np.array(y)
        self.class_priors_ = np.array([np.mean(y_arr == cl) for cl in self.classes_])
        
        if self.verbose:
            print("Final classes_ order:", self.classes_)
            print("class_priors_:", self.class_priors_)
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Remove target column if present.
        if self.target_col is not None:
            X = X.drop(columns=[self.target_col])
        
        X_llm, X_inner = self._prepare_data(X)
        X_inner = X_inner.reindex(columns=self.inner_columns_, fill_value=0)
        
        # Compute likelihoods from the inner estimator.
        if self.use_ml_prob_estimator:
            inner_probs = self.ml_prob_estimator.predict_proba(X_inner)
            # Apply smoothing to the ml estimator predictions using self.alpha.
            if self.alpha != 1.0:
                inner_probs = inner_probs ** self.alpha
                inner_probs = inner_probs / inner_probs.sum(axis=1, keepdims=True)
        else:
            num_instances = X_inner.shape[0]
            num_classes = len(self.classes_)
            inner_probs = np.full((num_instances, num_classes), 1 / num_classes)
        
        num_instances, num_classes = inner_probs.shape
        
        # Compute the pretrained prior from the LLM.
        if self.use_pretrained_llm_prob_estimator:
            pretrained_priors = self.pretrained_llm_prob_estimator.predict_proba(X_llm, list(self.classes_))
        else:
            uniform_prior = np.ones(len(self.class_priors_)) / len(self.class_priors_)
            pretrained_priors = np.tile(uniform_prior, (num_instances, 1))
        
        # Compute the combined probabilities.
        raw_combined = inner_probs * pretrained_priors * self.class_priors_
        row_sums = raw_combined.sum(axis=1, keepdims=True)
        fallback = inner_probs * self.class_priors_
        
        mask = (row_sums.flatten() != 0)
        combined_final = np.empty_like(raw_combined)
        combined_final[mask] = raw_combined[mask] / row_sums.flatten()[mask, None]
        combined_final[~mask] = fallback[~mask]
        
        if self.verbose:
            class_priors_str = " ".join([f"{p:.2f}" for p in self.class_priors_])
            for i in range(num_instances):
                pretrained_str = " ".join([f"{p:.2f}" for p in pretrained_priors[i]])
                likelihood_str = " ".join([f"{p:.2f}" for p in inner_probs[i]])
                combined_str = " ".join([f"{p:.2f}" for p in combined_final[i]])
                print(f"Instance {i}: likelihood: [{likelihood_str}]   "
                      f"pretrained_prior: [{pretrained_str}]   class_priors: [{class_priors_str}]   "
                      f"combined: [{combined_str}]")
        
        return combined_final
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]
