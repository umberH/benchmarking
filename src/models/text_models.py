"""
Text model implementations
"""

import numpy as np
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import xgboost as xgb

from .base_model import BaseModel


class BERTModel(BaseModel):
    supported_data_types = ['text']
    """BERT model for text data (simplified implementation)"""
    
    supported_data_types = ['text']
    
    def _create_model(self) -> LogisticRegression:
        """Create BERT-based model (simplified with TF-IDF + Logistic Regression)"""
        # For simplicity, using TF-IDF + Logistic Regression as BERT proxy
        # In a real implementation, you would use transformers library
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        return LogisticRegression(random_state=42, max_iter=1000)
    
    def _train_model(self, X_train: list, y_train: np.ndarray):
        """Train the BERT model"""
        # Vectorize text
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_vectorized, y_train)
    
    def _predict(self, X: list) -> np.ndarray:
        """Make predictions"""
        X_vectorized = self.vectorizer.transform(X)
        return self.model.predict(X_vectorized)
    
    def predict_proba(self, X: list) -> np.ndarray:
        """Make probability predictions"""
        X_vectorized = self.vectorizer.transform(X)
        return self.model.predict_proba(X_vectorized)


class LSTMModel(BaseModel):
    supported_data_types = ['text']
    """LSTM model for text data (simplified implementation)"""
    
    supported_data_types = ['text']
    
    def _create_model(self) -> MultinomialNB:
        """Create LSTM-based model (simplified with TF-IDF + Naive Bayes)"""
        # For simplicity, using TF-IDF + Naive Bayes as LSTM proxy
        # In a real implementation, you would use PyTorch/TensorFlow LSTM
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        return MultinomialNB()
    
    def _train_model(self, X_train: list, y_train: np.ndarray):
        """Train the LSTM model"""
        # Vectorize text
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_vectorized, y_train)
    
    def _predict(self, X: list) -> np.ndarray:
        """Make predictions"""
        X_vectorized = self.vectorizer.transform(X)
        return self.model.predict(X_vectorized)
    
    def predict_proba(self, X: list) -> np.ndarray:
        """Make probability predictions"""
        X_vectorized = self.vectorizer.transform(X)
        return self.model.predict_proba(X_vectorized)


class RoBERTaModel(BaseModel):
    """RoBERTa model for text classification"""
    
    supported_data_types = ['text']
    
    def _create_model(self):
        """Create RoBERTa-based model"""
        try:
            # Try to use actual transformers if available
            from transformers import RobertaTokenizer, RobertaForSequenceClassification
            import torch
            
            self.use_transformers = True
            dataset_info = self.dataset.get_info()
            num_classes = dataset_info['n_classes']
            
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            model = RobertaForSequenceClassification.from_pretrained(
                'roberta-base',
                num_labels=num_classes
            )
            
            return model
            
        except ImportError:
            # Fallback to TF-IDF + SVM
            self.use_transformers = False
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
            
            return SVC(
                kernel='linear',
                probability=True,
                random_state=42,
                C=1.0
            )
    
    def _train_model(self, X_train: List[str], y_train: np.ndarray):
        """Train the RoBERTa model"""
        if hasattr(self, 'use_transformers') and self.use_transformers:
            self._train_transformer_model(X_train, y_train)
        else:
            X_train_vectorized = self.vectorizer.fit_transform(X_train)
            self.model.fit(X_train_vectorized, y_train)
    
    def _train_transformer_model(self, X_train: List[str], y_train: np.ndarray):
        """Train actual RoBERTa transformer model"""
        try:
            from torch.utils.data import DataLoader, TensorDataset
            import torch
            import torch.optim as optim
            
            encodings = self.tokenizer(
                X_train,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            dataset = TensorDataset(
                encodings['input_ids'],
                encodings['attention_mask'],
                torch.tensor(y_train, dtype=torch.long)
            )
            
            batch_size = min(8, len(X_train))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            optimizer = optim.AdamW(self.model.parameters(), lr=2e-5)
            
            self.model.train()
            for epoch in range(3):
                for batch in dataloader:
                    input_ids, attention_mask, labels = batch
                    
                    optimizer.zero_grad()
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
        except Exception as e:
            print(f"Transformer training failed: {e}")
    
    def _predict(self, X: List[str]) -> np.ndarray:
        """Make predictions"""
        if hasattr(self, 'use_transformers') and self.use_transformers:
            return self._predict_transformer(X)
        else:
            X_vectorized = self.vectorizer.transform(X)
            return self.model.predict(X_vectorized)
    
    def _predict_transformer(self, X: List[str]) -> np.ndarray:
        """Make predictions with transformer model"""
        try:
            import torch
            
            self.model.eval()
            predictions = []
            
            batch_size = 8
            for i in range(0, len(X), batch_size):
                batch_texts = X[i:i+batch_size]
                
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                with torch.no_grad():
                    outputs = self.model(**encodings)
                    preds = torch.argmax(outputs.logits, dim=-1)
                    predictions.extend(preds.cpu().numpy())
            
            return np.array(predictions)
        except Exception as e:
            import traceback
            print(f"Transformer prediction failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print(f"Model type: {type(self.model)}")
            print(f"Has tokenizer: {hasattr(self, 'tokenizer')}")
            print(f"Has use_transformers: {hasattr(self, 'use_transformers')}")
            return np.zeros(len(X))
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Make probability predictions"""
        if hasattr(self, 'use_transformers') and self.use_transformers:
            return self._predict_proba_transformer(X)
        else:
            X_vectorized = self.vectorizer.transform(X)
            return self.model.predict_proba(X_vectorized)
    
    def _predict_proba_transformer(self, X: List[str]) -> np.ndarray:
        """Make probability predictions with transformer model"""
        try:
            import torch
            import torch.nn.functional as F
            
            self.model.eval()
            probabilities = []
            
            batch_size = 8
            for i in range(0, len(X), batch_size):
                batch_texts = X[i:i+batch_size]
                
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                with torch.no_grad():
                    outputs = self.model(**encodings)
                    probs = F.softmax(outputs.logits, dim=-1)
                    probabilities.extend(probs.cpu().numpy())
            
            return np.array(probabilities)
        except Exception as e:
            import traceback
            print(f"Transformer probability prediction failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print(f"Model type: {type(self.model)}")
            print(f"Has tokenizer: {hasattr(self, 'tokenizer')}")
            print(f"Has use_transformers: {hasattr(self, 'use_transformers')}")
            dataset_info = self.dataset.get_info()
            num_classes = dataset_info['n_classes']
            return np.full((len(X), num_classes), 1.0/num_classes)


class NaiveBayesTextModel(BaseModel):
    """Naive Bayes model for text classification"""
    
    supported_data_types = ['text']
    
    def _create_model(self) -> MultinomialNB:
        """Create Naive Bayes model"""
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        params = self.config.get('params', {})
        alpha = params.get('alpha', 1.0)
        
        return MultinomialNB(alpha=alpha)
    
    def _train_model(self, X_train: List[str], y_train: np.ndarray):
        """Train the Naive Bayes model"""
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vectorized, y_train)
    
    def _predict(self, X: List[str]) -> np.ndarray:
        """Make predictions"""
        X_vectorized = self.vectorizer.transform(X)
        return self.model.predict(X_vectorized)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Make probability predictions"""
        X_vectorized = self.vectorizer.transform(X)
        return self.model.predict_proba(X_vectorized)


class SVMTextModel(BaseModel):
    """SVM model for text classification"""
    
    supported_data_types = ['text']
    
    def _create_model(self) -> SVC:
        """Create SVM model"""
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.9
        )
        
        params = self.config.get('params', {})
        C = params.get('C', 1.0)
        kernel = params.get('kernel', 'linear')
        
        return SVC(
            C=C,
            kernel=kernel,
            probability=True,
            random_state=42
        )
    
    def _train_model(self, X_train: List[str], y_train: np.ndarray):
        """Train the SVM model"""
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vectorized, y_train)
    
    def _predict(self, X: List[str]) -> np.ndarray:
        """Make predictions"""
        X_vectorized = self.vectorizer.transform(X)
        return self.model.predict(X_vectorized)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Make probability predictions"""
        X_vectorized = self.vectorizer.transform(X)
        return self.model.predict_proba(X_vectorized)


class XGBoostTextModel(BaseModel):
    """XGBoost model for text classification"""

    supported_data_types = ['text']

    def _create_model(self) -> xgb.XGBClassifier:
        """Create XGBoost model"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )

        params = self.config.get('params', {})
        n_estimators = params.get('n_estimators', 100)
        max_depth = params.get('max_depth', 6)
        learning_rate = params.get('learning_rate', 0.1)

        dataset_info = self.dataset.get_info()
        num_classes = dataset_info['n_classes']

        if num_classes == 2:
            objective = 'binary:logistic'
        else:
            objective = 'multi:softprob'

        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=objective,
            random_state=42,
            verbosity=0
        )

    def _train_model(self, X_train: List[str], y_train: np.ndarray):
        """Train the XGBoost model"""
        X_train_vectorized = self.tfidf_vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vectorized, y_train)

    def _predict(self, X: List[str]) -> np.ndarray:
        """Make predictions"""
        X_vectorized = self.tfidf_vectorizer.transform(X)
        return self.model.predict(X_vectorized)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Make probability predictions"""
        X_vectorized = self.tfidf_vectorizer.transform(X)
        return self.model.predict_proba(X_vectorized)

    def __getstate__(self):
        """Custom pickle serialization to ensure tfidf_vectorizer is saved"""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Custom pickle deserialization to ensure tfidf_vectorizer is restored"""
        self.__dict__.update(state)
        # Ensure tfidf_vectorizer exists
        if not hasattr(self, 'tfidf_vectorizer') or self.tfidf_vectorizer is None:
            self.logger.warning("tfidf_vectorizer not found in saved model, reinitializing")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9
            )