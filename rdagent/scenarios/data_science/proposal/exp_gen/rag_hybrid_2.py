import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import pickle
import os
from datetime import datetime
import re
from collections import Counter
import hashlib
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import normalize
import builtins  # Add this import to ensure access to built-in functions
from rdagent.core.exception import CoderError
from rdagent.utils.agent.tpl import T  


logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    logger.warning("rank_bm25 not installed. BM25 functionality will be limited.")
    BM25Okapi = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not installed. Using numpy for vector search.")
    FAISS_AVAILABLE = False


try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not installed. Using fallback tokenization.")


class HybridRAGSystem:
    """
    Enhanced Hybrid RAG system using BM25 + Vector similarity for hypothesis generation
    Optimized for problem-method-context structure
    """
    
    def __init__(self, api_backend, cache_dir: str = "./rag_cache"):
        self.api_backend = api_backend
        self.cache_dir = cache_dir
        
        # Weights configuration
        self.semantic_weight = 0.7
        self.keyword_weight = 0.3
        
        # Field weights for BM25
        self.field_weights = {
            'problem': 1.0,   
            'method': 0.7,    
            'context': 0.5    
        }
        
        # Performance settings
        self.batch_size = 32
        self.max_workers = 4
        self.embedding_dim = 1536  
        
        # Disable parallel processing if SQLite cache is detected
        # to avoid thread safety issues
        self.enable_parallel = False  # Set to False due to SQLite thread safety
        
        # Caching
        self.query_cache = {}
        self.embedding_cache = {}
        self.max_cache_size = 10000
        self.cache_ttl = 3600 * 24 * 7  # 7 days
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize spaCy model if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Using fallback tokenization.")
                self.nlp = None
        else:
            self.nlp = None
        
        # Cache paths
        self.idea_pool_path = os.path.join(cache_dir, "idea_pool.json")
        self.embeddings_path = os.path.join(cache_dir, "idea_embeddings.pkl")
        self.bm25_path = os.path.join(cache_dir, "bm25_index.pkl")
        self.metadata_path = os.path.join(cache_dir, "metadata.json")
        self.faiss_index_path = os.path.join(cache_dir, "faiss.index")
        
        # Initialize data structures
        self.idea_pool = {}
        self.idea_embeddings = {}
        self.bm25_index = None
        self.tokenized_corpus = []
        self.idea_keys = []
        self.faiss_index = None
        
        # Domain-specific synonyms for essay scoring and ML

        # TODO  怎么强化？
        self.synonyms = {
    # 关于效率、速度和性能
    'efficiency': [
        'efficiency', 'speed', 'performance', 'fast', 'quick', 'optimization',
        'runtime', 'latency', 'throughput', 'computation time', 'acceleration',
        'optimized', 'efficient', 'performant', 'scalability'
    ],

    # 关于批处理
    'batch': [
        'batch', 'batching', 'batch size', 'mini-batch', 'micro-batch',
        'chunk', 'data chunking', 'grouping'
    ],

    # 关于填充和序列长度
    'padding': [
        'padding', 'pad', 'sequence length', 'variable length', 'fixed length',
        'truncation', 'truncate', 'max length', 'dynamic padding'
    ],

    # 关于模型集成
    'ensemble': [
        'ensemble', 'ensembling', 'combine', 'blend', 'average', 'voting',
        'stacking', 'bagging', 'boosting', 'model fusion', 'model aggregation',
        'mixture of experts'
    ],

    # 关于交叉验证
    'cross-validation': [
        'cross-validation', 'cv', 'k-fold', 'validation', 'evaluation strategy',
        'stratified k-fold', 'leave-one-out', 'train-test split', 'holdout set',
        'validation set'
    ],

    # 关于过拟合与泛化
    'overfitting': [
        'overfitting', 'overfit', 'generalization', 'regularization', 'underfitting',
        'generalization error', 'model complexity', 'dropout', 'l1 regularization',
        'l2 regularization', 'weight decay'
    ],

    # 关于方差和稳定性
    'variance': [
        'variance', 'variability', 'stability', 'consistency', 'robustness',
        'fluctuation', 'model stability', 'prediction variance', 'seed sensitivity'
    ],

    # 关于模型本身
    'model': [
        'model', 'algorithm', 'architecture', 'network', 'structure', 'framework',
        'classifier', 'regressor', 'neural network', 'deep learning model'
    ],

    # 关于推理和预测
    'inference': [
        'inference', 'prediction', 'test', 'evaluation', 'scoring', 'deployment',
        'serving', 'predict time', 'run model', 'apply model'
    ],

    # 关于评估指标
    'score': [
        'score', 'scoring', 'metric', 'performance', 'accuracy', 'error',
        'evaluation metric', 'loss', 'objective function', 'f1-score', 'precision',
        'recall', 'auc', 'rmse', 'mae' # 加入了一些具体指标
    ],

    # 关于文本数据
    'text': [
        'text', 'essay', 'document', 'sequence', 'input', 'corpus', 'string',
        'natural language', 'prose', 'utterance', 'paragraph'
    ],

    # 关于分词
    'tokenizer': [
        'tokenizer', 'tokenize', 'tokenization', 'encoding', 'vocabulary', 'vocab',
        'subword', 'wordpiece', 'bpe', 'sentencepiece', 'vectorization'
    ],

    # 关于特征工程
    'feature_engineering': [
        'feature engineering', 'feature extraction', 'feature selection',
        'feature creation', 'data preprocessing', 'feature transformation',
        'dimensionality reduction', 'pca', 'one-hot encoding'
    ],

    # 关于超参数调优
    'hyperparameter_tuning': [
        'hyperparameter tuning', 'hpo', 'tuning', 'optimization', 'grid search',
        'random search', 'bayesian optimization', 'optuna', 'hyperopt'
    ],

    # 关于模型训练
    'training': [
        'training', 'fitting', 'learning', 'optimization', 'backpropagation',
        'gradient descent', 'fine-tuning', 'pre-training', 'train loop'
    ],

    # 关于数据处理
    'data_processing': [
        'data processing', 'preprocessing', 'data cleaning', 'normalization',
        'standardization', 'data augmentation', 'feature scaling', 'data pipeline'
    ],

    # 关于模型架构
    'architecture': [
        'architecture', 'model structure', 'network design', 'layer configuration',
        'transformer', 'cnn', 'rnn', 'lstm', 'gru', 'attention mechanism'
    ],

    # 关于优化策略
    'optimization': [
        'optimization', 'optimizer', 'adam', 'sgd', 'learning rate', 'scheduler',
        'momentum', 'weight decay', 'gradient clipping', 'early stopping'
    ]
}
        
        # Load cached data
        self._load_cache()
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        if 'timestamp' not in cache_entry:
            return False
        return time.time() - cache_entry['timestamp'] < self.cache_ttl
    
    @lru_cache(maxsize=10000)
    def _tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization with spaCy support"""
        if SPACY_AVAILABLE and hasattr(self, 'nlp') and self.nlp is not None:
            return self._tokenize_spacy(text)
        else:
            return self._tokenize_enhanced_regex_v2(text)
    
    
    def _tokenize_spacy(self, text: str) -> List[str]:
        """spaCy-based tokenization"""

        protected_text = self._protect_technical_terms(text)
        doc = self.nlp(protected_text.lower())
        tokens = []
        
        technical_vocab = self._get_technical_vocab()
        
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                not token.is_space and 
                not token.like_num and
                len(token.text.strip()) > 1 and
                not token.text.isdigit()):
                
                # 处理技术术语
                if '_' in token.text:
                    tokens.append(token.text.replace('_', '-'))
                else:
                    # 对重要技术词汇保持原形，其他使用词形还原
                    if token.text.lower() in technical_vocab:
                        tokens.append(token.text.lower())
                    else:
                        tokens.append(token.lemma_)
        
        return tokens
    
    def _tokenize_enhanced_regex_v2(self, text: str) -> List[str]:
        """增强版正则表达式分词"""

        
        # 保护技术术语
        text = self._protect_technical_terms(text)
        
        # 提取tokens
        tokens = re.findall(r'\b\w+(?:_\w+)*\b', text.lower())
        
        # 后处理和过滤
        processed_tokens = []
        stop_words = self._get_extended_stop_words()
        
        for token in tokens:
            if (len(token) > 1 and 
                token not in stop_words and
                not token.isdigit() and
                not re.match(r'^\d+$', token)):
                
                # 恢复技术术语格式
                if '_' in token:
                    processed_tokens.append(token.replace('_', '-'))
                else:
                    processed_tokens.append(token)
        
        return processed_tokens
    
    def _protect_technical_terms(self, text: str) -> str:
        """保护技术术语不被错误分词"""
        patterns = [
            # 版本号和模型名
            (r'\b([a-zA-Z]+)-([a-zA-Z0-9.]+)\b', r'\1_\2'),
            (r'\b([a-zA-Z]+)(\d+(?:\.\d+)*)\b', r'\1_\2'),
            # 文件扩展名
            (r'\.(py|json|csv|txt|md|pkl|h5|pth)\b', r'_\1_file'),
            # 编程术语
            (r'\b(\w+)\.(\w+)\b', r'\1_\2'),
            # 缩写词
            (r'\b([A-Z]{2,})\b', lambda m: m.group(1).lower() + '_acronym'),
            # 特殊符号连接的术语
            (r'\b(\w+)-(\w+)\b', r'\1_\2'),
        ]
        
        for pattern, replacement in patterns:
            if callable(replacement):
                text = re.sub(pattern, replacement, text)
            else:
                text = re.sub(pattern, replacement, text)
        
        return text
    
    def _get_technical_vocab(self) -> set:
        """获取应该保持原形的技术词汇"""
        return {
            # 模型和算法
            'bert', 'gpt', 'transformer', 'lstm', 'cnn', 'rnn', 'gan', 'vae',
            'resnet', 'densenet', 'mobilenet', 'efficientnet', 'yolo',
            # 框架和库
            'pytorch', 'tensorflow', 'sklearn', 'pandas', 'numpy', 'scipy',
            'matplotlib', 'seaborn', 'plotly', 'keras', 'xgboost', 'lightgbm',
            # 评估指标
            'accuracy', 'precision', 'recall', 'f1', 'auc', 'roc', 'mae', 'mse',
            'rmse', 'r2', 'iou', 'dice', 'bleu', 'rouge',
            # 技术概念
            'overfitting', 'underfitting', 'regularization', 'normalization',
            'hyperparameter', 'embedding', 'tokenization', 'preprocessing',
            'augmentation', 'ensemble', 'bagging', 'boosting', 'stacking',
            # 数据相关
            'dataset', 'dataloader', 'batch', 'epoch', 'iteration', 'gradient',
            'backpropagation', 'optimization', 'learning_rate', 'momentum'
        }
    
    def _get_extended_stop_words(self) -> set:
        """扩展的停用词列表"""
        try:
            from spacy.lang.en.stop_words import STOP_WORDS
            return STOP_WORDS
        except ImportError:
            logger.warning("spaCy not installed. Using fallback stop words.")
            return {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'being', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
                'very', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
                'so', 'than', 'too', 'just', 'now', 'here', 'there', 'when', 'where',
                'why', 'how', 'all', 'any', 'both', 'each', 'few', 'no', 'nor', 'not',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                'my', 'your', 'his', 'her', 'its', 'our', 'their', 'myself', 'yourself',
                'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves'
            }


    # def _expand_query_with_synonyms(self, tokens: List[str]) -> List[str]:
    #     """Expand query tokens with domain-specific synonyms"""
    #     expanded_tokens = []
    #     seen = set()
        
    #     for token in tokens:
    #         if token not in seen:
    #             expanded_tokens.append(token)
    #             seen.add(token)
            
    #         # Check for synonyms
    #         for key, synonyms in self.synonyms.items():
    #             if token.lower() in [s.lower() for s in synonyms]:
    #                 for syn in synonyms:
    #                     syn_tokens = syn.split()
    #                     for syn_token in syn_tokens:
    #                         if syn_token.lower() not in seen:
    #                             expanded_tokens.append(syn_token.lower())
    #                             seen.add(syn_token.lower())
        
    #     return expanded_tokens
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract top keywords from text based on frequency"""
        tokens = self._tokenize(text)
        
        # Enhanced stop words for technical content
        stop_words=self._get_extended_stop_words()
        
        # 过滤停用词
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        
        # Get top frequent words
        word_freq = Counter(tokens)
        return [word for word, _ in word_freq.most_common(top_k)]
    
    def _get_idea_text(self, idea_data: Any) -> str:
        """Extract text from idea data for indexing"""
        if isinstance(idea_data, dict):
            parts = []
            # 按重要性顺序提取文本
            for field in ['problem', 'method', 'context']:
                if field in idea_data and idea_data[field]:
                    parts.append(str(idea_data[field]))
            return " ".join(parts)
        else:
            return str(idea_data)
    
    def _load_cache(self):
        """Load cached idea pool and indices"""
        try:
            # Load idea pool
            if os.path.exists(self.idea_pool_path):
                with builtins.open(self.idea_pool_path, 'r', encoding='utf-8') as f:
                    self.idea_pool = json.load(f)
                logger.info(f"Loaded {len(self.idea_pool)} ideas from cache")
            
            # Load embeddings
            if os.path.exists(self.embeddings_path):
                with builtins.open(self.embeddings_path, 'rb') as f:
                    self.idea_embeddings = pickle.load(f)
                logger.info(f"Loaded embeddings for {len(self.idea_embeddings)} ideas")
            
            # Load BM25 index
            if os.path.exists(self.bm25_path) and BM25Okapi:
                with builtins.open(self.bm25_path, 'rb') as f:
                    bm25_data = pickle.load(f)
                    self.bm25_index = bm25_data['bm25']
                    self.tokenized_corpus = bm25_data['corpus']
                    self.idea_keys = bm25_data['keys']
                logger.info("Loaded BM25 index from cache")
            
            # Load FAISS index if available
            if FAISS_AVAILABLE and os.path.exists(self.faiss_index_path):
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                logger.info("Loaded FAISS index from cache")
            
            # Load metadata
            if os.path.exists(self.metadata_path):
                with builtins.open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    logger.info(f"Cache last updated: {metadata.get('last_updated', 'Unknown')}")
                    
        except Exception as e:
            logger.warning(f"Error loading cache: {e}. Starting with empty cache.")
    
    def _save_cache(self):
        """Save idea pool and indices to cache"""
        try:
            # Save idea pool
            with builtins.open(self.idea_pool_path, 'w', encoding='utf-8') as f:
                json.dump(self.idea_pool, f, indent=2, ensure_ascii=False)
            
            # Save embeddings
            with builtins.open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.idea_embeddings, f)
            
            # Save BM25 index
            if self.bm25_index and BM25Okapi:
                with builtins.open(self.bm25_path, 'wb') as f:
                    pickle.dump({
                        'bm25': self.bm25_index,
                        'corpus': self.tokenized_corpus,
                        'keys': self.idea_keys
                    }, f)
            
            # Save FAISS index
            if FAISS_AVAILABLE and self.faiss_index:
                faiss.write_index(self.faiss_index, self.faiss_index_path)
            
            # Save metadata
            metadata = {
                'last_updated': datetime.now().isoformat(),
                'total_ideas': len(self.idea_pool),
                'semantic_weight': self.semantic_weight,
                'keyword_weight': self.keyword_weight,
                'faiss_enabled': FAISS_AVAILABLE and self.faiss_index is not None
            }
            with builtins.open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info("Successfully saved cache")
            
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def batch_create_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Batch create embeddings with caching"""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            
            if cache_key in self.embedding_cache:
                cache_entry = self.embedding_cache[cache_key]
                if self._is_cache_valid(cache_entry):
                    embeddings.append(cache_entry['embedding'])
                    continue
            
            embeddings.append(None)
            uncached_texts.append(text)
            uncached_indices.append(i)
        
        # Batch process uncached texts
        if uncached_texts:
            for batch_start in range(0, len(uncached_texts), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(uncached_texts))
                batch_texts = uncached_texts[batch_start:batch_end]
                
                try:
                    # Call API
                    batch_embeddings = self.api_backend.create_embedding(batch_texts)
                    
                    # Update results and cache
                    for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                        idx = uncached_indices[batch_start + j]
                        embeddings[idx] = np.array(embedding)
                        
                        # Update cache
                        cache_key = self._get_cache_key(text)
                        self.embedding_cache[cache_key] = {
                            'embedding': embeddings[idx],
                            'timestamp': time.time()
                        }
                        
                except Exception as e:
                    logger.error(f"Batch embedding creation failed: {e}")
                    # Fallback to individual requests
                    for j, text in enumerate(batch_texts):
                        idx = uncached_indices[batch_start + j]
                        try:
                            embedding = self.api_backend.create_embedding([text])[0]
                            embeddings[idx] = np.array(embedding)
                        except:
                            embeddings[idx] = np.zeros(self.embedding_dim)
        
        # Clean up old cache entries periodically
        if len(self.embedding_cache) > self.max_cache_size:
            self._cleanup_embedding_cache()
        
        return embeddings
    
    def _cleanup_embedding_cache(self):
        """Clean up old embedding cache entries"""
        # Remove expired entries
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.embedding_cache.items()
            if not self._is_cache_valid(entry)
        ]
        for key in expired_keys:
            del self.embedding_cache[key]
        
        # If still too large, remove oldest entries
        if len(self.embedding_cache) > self.max_cache_size:
            sorted_entries = sorted(
                self.embedding_cache.items(),
                key=lambda x: x[1].get('timestamp', 0)
            )
            # Keep only the newest entries
            self.embedding_cache = dict(sorted_entries[-self.max_cache_size:])
    
    def add_ideas(self, new_ideas: Dict, update_cache: bool = True):
        """Add new ideas to the pool and update indices incrementally"""
        if not new_ideas:
            return
            
        logger.info(f"Adding {len(new_ideas)} new ideas to pool")
        
        # Track what's actually new
        actually_new = {}
        for idea_name, idea_data in new_ideas.items():
            if idea_name not in self.idea_pool:
                actually_new[idea_name] = idea_data
                self.idea_pool[idea_name] = idea_data
        
        if not actually_new:
            logger.info("No new ideas to add (all already exist)")
            return
        
        # Compute embeddings for new ideas
        new_texts = []
        new_names = []
        
        for idea_name, idea_data in actually_new.items():
            text = self._get_idea_text(idea_data)
            new_texts.append(text)
            new_names.append(idea_name)
        
        if new_texts:
            # Batch compute embeddings
            new_embeddings = self.batch_create_embeddings(new_texts)
            
            # Update embedding dictionary
            for name, embedding in zip(new_names, new_embeddings):
                self.idea_embeddings[name] = embedding
        
        # Rebuild indices
        self._rebuild_indices()
        
        if update_cache:
            self._save_cache()
    
    def _rebuild_indices(self):
        """Rebuild all indices (BM25 and FAISS)"""
        # Rebuild BM25 index
        self._rebuild_bm25_index()
        
        # Rebuild FAISS index if available
        if FAISS_AVAILABLE:
            self._rebuild_faiss_index()
    
    def _rebuild_bm25_index(self):
        """Rebuild BM25 index from current idea pool"""
        if not BM25Okapi:
            logger.warning("BM25 not available")
            return
            
        self.idea_keys = list(self.idea_pool.keys())
        self.tokenized_corpus = []
        
        for idea_name in self.idea_keys:
            idea_data = self.idea_pool[idea_name]
            text = self._get_idea_text(idea_data)
            tokens = self._tokenize(text)
            self.tokenized_corpus.append(tokens)
        
        # Create BM25 index
        if self.tokenized_corpus:
            self.bm25_index = BM25Okapi(self.tokenized_corpus)
            logger.info(f"Rebuilt BM25 index with {len(self.tokenized_corpus)} documents")
    
    def _rebuild_faiss_index(self):
        """Build or rebuild FAISS index for fast vector search"""
        if not self.idea_embeddings:
            return
        
        # Ensure idea_keys matches embeddings
        self.idea_keys = [k for k in self.idea_keys if k in self.idea_embeddings]
        
        # Collect all embeddings in order
        embeddings = []
        for idea_name in self.idea_keys:
            if idea_name in self.idea_embeddings:
                embeddings.append(self.idea_embeddings[idea_name])
        
        if not embeddings:
            return
        
        embeddings_matrix = np.array(embeddings).astype('float32')
        
        # Normalize for cosine similarity
        embeddings_matrix = normalize(embeddings_matrix, axis=1)
        
        n_vectors = embeddings_matrix.shape[0]
        
        if n_vectors < 1000:
            # Use flat index for small datasets
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            # Use IVF index for larger datasets
            nlist = min(100, n_vectors // 10)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.faiss_index.train(embeddings_matrix)
        
        self.faiss_index.add(embeddings_matrix)
        logger.info(f"Built FAISS index with {n_vectors} vectors")
    
    def _compute_bm25_scores(self, query: str) -> np.ndarray:
        """Compute BM25 scores for a query"""
        if not self.bm25_index or not BM25Okapi:
            return np.zeros(len(self.idea_keys))
        
        tokenized_query = self._tokenize(query)
        # expanded_query = self._expand_query_with_synonyms(tokenized_query)
        
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Normalize scores to [0, 1]
        max_score = scores.max() if scores.max() > 0 else 1.0
        normalized_scores = scores / max_score
        
        return normalized_scores
    
    def _compute_semantic_scores(self, query: str, top_k: int = None) -> np.ndarray:
        """Compute semantic similarity scores for a query"""
        # Get query embedding
        query_embedding = self.api_backend.create_embedding([query])[0]
        query_embedding = np.array(query_embedding).astype('float32')
        query_embedding = normalize(query_embedding.reshape(1, -1), axis=1)[0]
        
        if FAISS_AVAILABLE and self.faiss_index:
            # Use FAISS for fast search
            k = top_k or len(self.idea_keys)
            k = min(k, self.faiss_index.ntotal)
            
            distances, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1), k
            )
            
            # Convert to full score array 
            # TODO 能不能改进？
            scores = np.zeros(len(self.idea_keys))
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1:  # Valid index
                    scores[idx] = (dist + 1) / 2  # Convert to [0, 1]
            
            return scores
        else:
            # Fallback to numpy computation
            scores = []
            for idea_name in self.idea_keys:
                if idea_name in self.idea_embeddings:
                    idea_embedding = np.array(self.idea_embeddings[idea_name])
                    idea_embedding = normalize(idea_embedding.reshape(1, -1), axis=1)[0]
                    
                    # Cosine similarity
                    similarity = np.dot(query_embedding, idea_embedding)
                    scores.append((similarity + 1) / 2)  # Normalize to [0, 1]
                else:
                    scores.append(0.0)
            
            return np.array(scores)
    
    def _compute_field_bm25_scores(self, problem_data: Dict[str, str]) -> np.ndarray:
        """Compute BM25 scores considering field weights"""
        if not self.bm25_index or not BM25Okapi:
            return np.zeros(len(self.idea_keys))
        
        field_scores = []
        
        # TODO 这里的field_weights包括idea_pool的method,problem,context但是problem_data只有problem,reason,label,重合的只有problem
        for field, weight in self.field_weights.items():
            if field in problem_data and problem_data[field]:
                query = problem_data[field]
                scores = self._compute_bm25_scores(query)
                field_scores.append(scores * weight)
        
        if field_scores:
            # Combine field scores
            combined_scores = np.sum(field_scores, axis=0)
            # Normalize
            max_score = combined_scores.max() if combined_scores.max() > 0 else 1.0
            return combined_scores / max_score
        else:
            return np.zeros(len(self.idea_keys))
    
    def retrieve_relevant_ideas(
        self, 
        problem_name: str,
        problem_data: Dict[str, str],
        top_k: int = 5,
        min_score: float = 0.5,
        use_reranking: bool = False
    ) -> List[Dict]:
        """Enhanced retrieval with multi-field search and reranking"""
        # Build comprehensive query
        query_parts = []
        
        # TODO 是不是得改改拼接方式？

        # 从problem_data中提取query（主要是problem字段）
        if 'problem' in problem_data and problem_data['problem']:
            query_parts.append(problem_data['problem'])
        if 'reason' in problem_data and problem_data['reason']:
            query_parts.append(problem_data['reason'])
        
        if not query_parts:
            logger.warning(f"No valid query parts for problem {problem_name}")
            return []
        
        query = " ".join(query_parts)
        
        # Check query cache
        cache_key = self._get_cache_key(f"{problem_name}:{query}:{top_k}")
        if cache_key in self.query_cache:
            cache_entry = self.query_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                return cache_entry['results']
            

        adj_keyword_weight = self.keyword_weight
        adj_semantic_weight = 1 - adj_keyword_weight
        
        # Get BM25 scores (field-weighted)
        bm25_scores = self._compute_field_bm25_scores(problem_data)
        
        # Get semantic scores
        semantic_scores = self._compute_semantic_scores(query, top_k=top_k*3 if use_reranking else top_k)
        
        # Combine scores
        final_scores = (
            adj_semantic_weight * semantic_scores + 
            adj_keyword_weight * bm25_scores
        )
        
        # Get top candidates
        initial_top_k = min(top_k * 3, len(self.idea_keys)) if use_reranking else top_k
        top_indices = np.argsort(final_scores)[::-1][:initial_top_k]
        
        # Collect candidates
        candidates = []
        for idx in top_indices:
            score = final_scores[idx]
            if score >= min_score:  
                idea_name = self.idea_keys[idx]
                idea_data = self.idea_pool[idea_name]
                
                candidates.append({
                    'idea_name': idea_name,
                    'initial_score': float(score),
                    'semantic_score': float(semantic_scores[idx]),
                    'keyword_score': float(bm25_scores[idx]),
                    'idea_data': idea_data,
                    'idx': idx
                })
        
        # # Rerank if enabled
        # if use_reranking and candidates:
        #     candidates = self._rerank_candidates(
        #         query, problem_data, candidates, top_k
        #     )
        
        # Final selection
        final_results = []
        for candidate in candidates[:top_k]:
            if candidate.get('initial_score', 0) >= min_score:
                # Extract keywords
                idea_text = self._get_idea_text(candidate['idea_data'])
                keywords = self._extract_keywords(idea_text, top_k=5)

                result = {
                    'idea_name': candidate['idea_name'],
                    'similarity_score': float(candidate.get('initial_score', 0)),
                    'semantic_score': float(candidate['semantic_score']),
                    'keyword_score': float(candidate['keyword_score']),
                    'idea_data': candidate['idea_data'],
                    'key_concepts': keywords
                }
                
                final_results.append(result)
        
        # Cache results
        self.query_cache[cache_key] = {
            'results': final_results,
            'timestamp': time.time()
        }
        
        logger.info(f"Found {len(final_results)} relevant ideas for problem '{problem_name}'")
        return final_results

    
    def parallel_retrieve(
        self, 
        problems: Dict[str, Dict[str, str]], 
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        """Parallel retrieval for multiple problems"""
        results = {}
        
        if not self.enable_parallel or len(problems) == 1:
            # Sequential processing (thread-safe)
            for problem_name, problem_data in problems.items():
                try:
                    results[problem_name] = self.retrieve_relevant_ideas(
                        problem_name, problem_data, top_k
                    )
                except Exception as e:
                    logger.error(f"Retrieval failed for {problem_name}: {e}")
                    results[problem_name] = []
        else:
            # Parallel processing (use only if thread-safe)
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(problems))) as executor:
                future_to_problem = {
                    executor.submit(
                        self.retrieve_relevant_ideas,
                        problem_name,
                        problem_data,
                        top_k
                    ): problem_name
                    for problem_name, problem_data in problems.items()
                }
                
                for future in as_completed(future_to_problem):
                    problem_name = future_to_problem[future]
                    try:
                        relevant_ideas = future.result()
                        results[problem_name] = relevant_ideas
                    except Exception as e:
                        logger.error(f"Retrieval failed for {problem_name}: {e}")
                        results[problem_name] = []
        
        return results
    
    def _format_problems_with_ideas(
        self, 
        problems: Dict[str, Dict[str, str]], 
        problem_to_ideas: Dict[str, List[Dict]]
    ) -> str:
        """Format problems with their retrieved relevant ideas for hypothesis generation"""
        formatted_sections = []
        
        for index, (problem_name, problem_data) in enumerate(problems.items()):
            # 使用模板字符串构建结构化内容
            section = {
                "problem": {
                    "name": problem_name,
                    "description": problem_data.get('problem', ''),
                    "reason": problem_data.get('reason', ''),
                    "label": problem_data.get('label', ''),
                    "number": index + 1
                },
                "similar_cases": []
            }
            
            # 添加相关想法
            relevant_ideas = problem_to_ideas.get(problem_name, [])
            if relevant_ideas:
                for i, idea_info in enumerate(relevant_ideas[:3], 1):
                    case = {
                        "id": i,
                        "name": idea_info['idea_name'],
                        "relevance": round(idea_info['similarity_score'], 2),
                        "problem": idea_info['idea_data'].get('problem', ''),
                        "solution": idea_info['idea_data'].get('method', ''),
                        "context": idea_info['idea_data'].get('context', ''),
                        "concepts": idea_info.get('key_concepts', [])[:5]
                    }
                    section["similar_cases"].append(case)
            
            # 转换为Markdown格式
            formatted_sections.append(self._convert_to_markdown(section))
        
        return "\n".join(formatted_sections)
    
    def _convert_to_markdown(self, section: Dict) -> str:
        """Convert structured data to Markdown format"""
        md = []
        
        # Problem section
        md.append(f"# Problem Analysis for Problem:{section['problem']['number']}")
        md.append(f"## Problem Name: {section['problem']['name']}")
        md.append(f"## Problem Description: {section['problem']['description']}")
        if section['problem']['reason']:
            md.append(f"## Reason for Problem: {section['problem']['reason']}")
        if section['problem']['label']:
            md.append(f"## Label: {section['problem']['label']}")
        
        # Similar cases section
        if section['similar_cases']:
            md.append("## Similar Cases Analysis")
            md.append("### Top 3 Relevant Ideas and Solutions")
            for case in section['similar_cases']:
                md.append(f"{case['id']}. **{case['name']}** (Relevance: {case['relevance']})")
                md.append(f"   - **Problem Description**: {case['problem']}")
                md.append(f"   - **Method Description**: {case['solution']}")
                md.append(f"   - **Context Description**: {case['context']}")
                if case['concepts']:
                    md.append(f"   - **Key Concepts**: {', '.join(case['concepts'])}")
                md.append("")
        else:
            md.append("**Note**: No highly similar cases found in the knowledge base. Consider novel approaches.")
        
        return "\n".join(md)
    
    def hypothesis_draft(
        self,
        no_sota_idea_path: str | None,
        component_desc: str,
        scenario_desc: str,
        exp_and_feedback_list_desc: str,
        problems: Dict[str, Dict[str, str]],
    ) -> Dict:
        """
        Generate hypotheses using hybrid RAG approach
        
        This method:
        1. Loads or updates the idea pool
        2. Uses hybrid search to find relevant ideas
        3. Generates hypotheses informed by successful similar cases
        """
        logger.info("Starting hypothesis draft with Enhanced Hybrid RAG")
        
        # Load idea pool if provided or if pool is empty
        if no_sota_idea_path and os.path.exists(no_sota_idea_path):
            try:
                with builtins.open(no_sota_idea_path, 'r', encoding='utf-8') as f:
                    new_ideas = json.load(f)
                    self.add_ideas(new_ideas, update_cache=True)
                    logger.info(f"Loaded {len(new_ideas)} ideas from {no_sota_idea_path}")
            except Exception as e:
                logger.error(f"Failed to load ideas from {no_sota_idea_path}: {e}")
        
        # Check if we have any ideas in the pool
        if not self.idea_pool:
            logger.warning("No ideas in pool. Proceeding without historical references.")
        
        # Retrieve relevant ideas for each problem in parallel
        problem_to_ideas = self.parallel_retrieve(problems, top_k=5)
        
        # Format problems with matched ideas
        formatted_problems = self._format_problems_with_ideas(problems, problem_to_ideas)
        
        # Generate hypotheses using the formatted problems and ideas
        hypothesis_dict = self._generate_hypotheses_with_ideas(
            component_desc,scenario_desc, exp_and_feedback_list_desc, formatted_problems
        )
        
        # Enhance hypotheses with references
        # enhanced_hypotheses = self._enhance_hypotheses_with_references(
        #     hypothesis_dict, problem_to_ideas
        # )
        
        logger.info(f"Generated {len(hypothesis_dict)} hypotheses with enhanced hybrid RAG")
        return hypothesis_dict
    
    def _generate_hypotheses_with_ideas(
        self,
        component_desc: str,
        scenario_desc: str,
        exp_and_feedback_list_desc: str,
        formatted_problems: str
    ) -> Dict[str, Dict[str, Any]]:
        """Generate hypotheses using the formatted problems and ideas"""
        try:

            # Prepare prompts using template system
            sys_prompt = T(".prompts_v2:hypothesis_draft.system").r(
                hypothesis_spec=T(".prompts_v2:specification.hypothesis").r(pipeline=True),
                hypothesis_output_format=T(".prompts_v2:output_format.hypothesis").r(
                    pipeline=True,
                    enable_idea_pool=True
                ),
                component_desc=component_desc,
                enable_idea_pool=True
            )
            
            user_prompt = T(".prompts_v2:hypothesis_draft.user").r(
                scenario_desc=scenario_desc,
                exp_and_feedback_list_desc=exp_and_feedback_list_desc,
                problems=formatted_problems,
            )
            
            # Call LLM API to generate hypotheses
            response = self.api_backend.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                json_mode=True,
                json_target_type=Dict[str, Dict],
            )
            
            hypothesis_dict = json.loads(response)
            logger.info(f"Generated {len(hypothesis_dict)} hypotheses")
            return hypothesis_dict
            
        except Exception as e:
            error_msg = f"Failed to generate hypotheses: {e}"
            logger.warning(error_msg)
            # Return empty dict instead of raising to allow graceful degradation
            return {}
    
    # def _enhance_hypotheses_with_references(
    #     self,
    #     hypothesis_dict: Dict[str, Dict],
    #     problem_to_ideas: Dict[str, List[Dict]]
    # ) -> Dict[str, Dict]:
    #     """Enhance generated hypotheses with references to relevant ideas"""
    #     enhanced_hypotheses = {}
        
    #     for problem_name, hypothesis in hypothesis_dict.items():
    #         # Copy original hypothesis
    #         enhanced_hypothesis = hypothesis.copy()
            
    #         # Add references to relevant ideas
    #         relevant_ideas = problem_to_ideas.get(problem_name, [])
    #         if relevant_ideas:
    #             # Add top 3 supporting ideas
    #             supporting_refs = []
    #             for idea_info in relevant_ideas[:3]:
    #                 ref_info = {
    #                     'idea_name': idea_info['idea_name'],
    #                     'relevance_score': round(idea_info['similarity_score'], 3),
    #                     'key_concepts': idea_info.get('key_concepts', [])[:5]
    #                 }
                    
                    
    #                 # Add brief method summary
    #                 if isinstance(idea_info['idea_data'], dict) and 'method' in idea_info['idea_data']:
    #                     method_preview = idea_info['idea_data']['method'][:100] + "..."
    #                     ref_info['method_preview'] = method_preview
                    
    #                 supporting_refs.append(ref_info)
                
    #             enhanced_hypothesis['supporting_ideas'] = supporting_refs
    #             enhanced_hypothesis['evidence_based'] = True
                
    #             # Set confidence level based on top match score
    #             top_score = relevant_ideas[0]['similarity_score']
    #             if top_score > 0.8:
    #                 enhanced_hypothesis['confidence_level'] = 'very_high'
    #             elif top_score > 0.6:
    #                 enhanced_hypothesis['confidence_level'] = 'high'
    #             elif top_score > 0.4:
    #                 enhanced_hypothesis['confidence_level'] = 'medium'
    #             else:
    #                 enhanced_hypothesis['confidence_level'] = 'low'
                    
    #             # Add retrieval metadata
    #             enhanced_hypothesis['retrieval_metadata'] = {
    #                 'num_relevant_ideas': len(relevant_ideas),
    #                 'avg_relevance_score': round(
    #                     sum(idea['similarity_score'] for idea in relevant_ideas) / len(relevant_ideas), 
    #                     3
    #                 ),
    #                 'retrieval_timestamp': datetime.now().isoformat()
    #             }
    #         else:
    #             enhanced_hypothesis['evidence_based'] = False
    #             enhanced_hypothesis['confidence_level'] = 'exploratory'
    #             enhanced_hypothesis['note'] = 'No similar cases found - novel hypothesis'
            
    #         enhanced_hypotheses[problem_name] = enhanced_hypothesis
        
    #     return enhanced_hypotheses
    
    def update_weights(self, semantic_weight: float = None, field_weights: Dict[str, float] = None):
        """Update retrieval weights"""
        if semantic_weight is not None:
            self.semantic_weight = max(0.0, min(1.0, semantic_weight))
            self.keyword_weight = 1 - self.semantic_weight
            logger.info(f"Updated weights - Semantic: {self.semantic_weight}, Keyword: {self.keyword_weight}")
        
        if field_weights:
            self.field_weights.update(field_weights)
            logger.info(f"Updated field weights: {self.field_weights}")
    
    def set_parallel_processing(self, enable: bool):
        """Enable or disable parallel processing
        
        Args:
            enable: Whether to enable parallel processing.
                    Should be False when using SQLite cache to avoid thread safety issues.
        """
        self.enable_parallel = enable
        logger.info(f"Parallel processing {'enabled' if enable else 'disabled'}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            'total_ideas': len(self.idea_pool),
            'total_embeddings': len(self.idea_embeddings),
            'cache_size': {
                'query_cache': len(self.query_cache),
                'embedding_cache': len(self.embedding_cache)
            },
            'index_status': {
                'bm25': self.bm25_index is not None,
                'faiss': FAISS_AVAILABLE and self.faiss_index is not None
            },
            'weights': {
                'semantic': self.semantic_weight,
                'keyword': self.keyword_weight,
                'fields': self.field_weights
            },
            'parallel_processing': self.enable_parallel
        }
        
        if FAISS_AVAILABLE and self.faiss_index:
            stats['faiss_vectors'] = self.faiss_index.ntotal
        
        # Add field coverage statistics
        if self.idea_pool:
            field_coverage = {}
            for field in ['problem', 'method', 'context']:
                count = sum(1 for idea in self.idea_pool.values() 
                          if isinstance(idea, dict) and field in idea and idea[field])
                field_coverage[field] = f"{(count/len(self.idea_pool)*100):.1f}%"
            stats['field_coverage'] = field_coverage
        
        return stats
    
    def clear_cache(self):
        """Clear all cached data"""
        import shutil
        try:
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Reset all data structures
            self.idea_pool = {}
            self.idea_embeddings = {}
            self.bm25_index = None
            self.tokenized_corpus = []
            self.idea_keys = []
            self.faiss_index = None
            self.query_cache = {}
            self.embedding_cache = {}
            
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def export_ideas(self, output_path: str):
        """Export idea pool to file"""
        try:
            with builtins.open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.idea_pool, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported {len(self.idea_pool)} ideas to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export ideas: {e}")
    
    def analyze_idea_pool(self) -> Dict[str, Any]:
        """Analyze the current idea pool for insights"""
        if not self.idea_pool:
            return {"error": "No ideas in pool"}
        
        analysis = {
            'total_ideas': len(self.idea_pool),
            'avg_text_length': 0,
            'top_keywords': [],
            'method_complexity': {
                'simple': 0,  # < 50 words
                'medium': 0,  # 50-100 words
                'detailed': 0  # > 100 words
            }
        }
        
        all_keywords = Counter()
        total_length = 0
        
        for idea_name, idea_data in self.idea_pool.items():
            if isinstance(idea_data, dict):
                # Text length
                text = self._get_idea_text(idea_data)
                total_length += len(text)
                
                # Keywords
                keywords = self._extract_keywords(text, 20)
                all_keywords.update(keywords)
                
                # Method complexity
                if 'method' in idea_data:
                    method_words = len(idea_data['method'].split())
                    if method_words < 50:
                        analysis['method_complexity']['simple'] += 1
                    elif method_words < 100:
                        analysis['method_complexity']['medium'] += 1
                    else:
                        analysis['method_complexity']['detailed'] += 1
        
        analysis['avg_text_length'] = int(total_length / len(self.idea_pool))
        analysis['top_keywords'] = [word for word, _ in all_keywords.most_common(20)]
        
        return analysis
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        # Save cache if there are unsaved changes
        if hasattr(self, 'idea_pool') and self.idea_pool:
            try:
                # Check if builtins.open is still available
                if hasattr(builtins, 'open') and builtins.open is not None:
                    self._save_cache()
            except Exception as e:
                # During interpreter shutdown, even logger might not be available
                try:
                    logger.warning(f"Failed to save cache during cleanup: {e}")
                except:
                    pass


# Convenience class that matches the original interface
class HypothesisRAGSystem(HybridRAGSystem):
    """Alias for backward compatibility"""
    pass