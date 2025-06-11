import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
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
import builtins
from rdagent.core.exception import CoderError
from rdagent.utils.agent.tpl import T
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize 


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


# ============ 新增：增强的分词器类 ============
class EnhancedTokenizer:
    """增强的分词器，支持 n-gram 和技术短语提取"""
    
    def __init__(self):
        # 尝试下载 NLTK 数据
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
        
        # 技术领域的常见短语
        self.technical_phrases = {
            # ML/DL 术语
            'neural network', 'deep learning', 'machine learning',
            'gradient descent', 'back propagation', 'batch normalization',
            'learning rate', 'weight decay', 'early stopping',
            'cross validation', 'k fold', 'train test split',
            
            # 性能相关
            'inference time', 'training time', 'memory usage',
            'computational cost', 'time complexity', 'space complexity',
            
            # 技术组件
            'attention mechanism', 'transformer model', 'embedding layer',
            'output layer', 'hidden layer', 'activation function',
            
            # 数据处理
            'data augmentation', 'feature engineering', 'data preprocessing',
            'batch processing', 'sequence padding', 'data pipeline',
            
            # 问题描述
            'overfitting problem', 'underfitting issue', 'convergence issue',
            'memory overflow', 'gradient explosion', 'vanishing gradient',
            
            # 评估相关
            'evaluation metric', 'performance metric', 'loss function',
            'accuracy score', 'f1 score', 'precision recall',
            
            # 优化相关
            'hyperparameter tuning', 'grid search', 'random search',
            'bayesian optimization', 'early stopping', 'learning rate decay'
        }
        
        # 重要的技术缩写
        self.abbreviations = {
            'lr': 'learning rate',
            'cv': 'cross validation',
            'nn': 'neural network',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'gpu': 'graphics processing unit',
            'cpu': 'central processing unit',
            'api': 'application programming interface',
            'rnn': 'recurrent neural network',
            'cnn': 'convolutional neural network',
            'lstm': 'long short term memory',
            'bert': 'bidirectional encoder representations from transformers',
            'gpt': 'generative pretrained transformer'
        }
    
    def extract_phrases(self, text: str) -> List[str]:
        """提取技术短语"""
        text_lower = text.lower()
        found_phrases = []
        
        # 查找预定义的技术短语
        for phrase in self.technical_phrases:
            if phrase in text_lower:
                found_phrases.append(phrase)
                # 将找到的短语替换为占位符，避免重复提取
                text_lower = text_lower.replace(phrase, f"__PHRASE_{len(found_phrases)}__")
        
        return found_phrases
    
    def tokenize_with_ngrams(self, text: str, n_range=(1, 3)) -> List[str]:
        """
        生成包含 n-gram 的 tokens
        
        Args:
            text: 输入文本
            n_range: n-gram 范围，(min_n, max_n)
        """
        # 1. 先提取技术短语
        phrases = self.extract_phrases(text)
        
        # 2. 基础分词
        text_lower = text.lower()
        
        # 替换短语为特殊标记
        for i, phrase in enumerate(phrases):
            text_lower = text_lower.replace(phrase, f"__PHRASE_{i}__")
        
        # 3. 分词（保留数字和连字符词）
        tokens = re.findall(r'\b(?:\w+(?:-\w+)*|\d+(?:\.\d+)?)\b', text_lower)
        
        # 4. 扩展缩写
        expanded_tokens = []
        for token in tokens:
            if token in self.abbreviations:
                # 添加缩写的扩展形式
                expanded_tokens.extend(self.abbreviations[token].split())
            if not token.startswith("__PHRASE_"):  # 不添加占位符
                expanded_tokens.append(token)
        
        # 5. 生成 n-grams
        all_tokens = []
        
        # 添加原始 tokens (1-gram)
        all_tokens.extend(expanded_tokens)
        
        # 生成 2-gram 到 max_n-gram
        for n in range(2, n_range[1] + 1):
            if len(expanded_tokens) >= n:
                n_grams = list(ngrams(expanded_tokens, n))
                # 将 n-gram 转换为字符串
                all_tokens.extend(['_'.join(gram) for gram in n_grams])
        
        # 6. 添加原始短语
        all_tokens.extend(phrases)
        
        return all_tokens



# ============ 新增：增强的 BM25 类 ============
class EnhancedBM25:
    """增强的 BM25，支持 n-gram 和短语"""
    
    def __init__(self, corpus: List[str], tokenizer: EnhancedTokenizer = None):
        self.tokenizer = tokenizer or EnhancedTokenizer()
        self.corpus = corpus
        self.corpus_size = len(corpus)
        
        # 分词并构建索引
        self.tokenized_corpus = []
        self.doc_lengths = []
        self.df = {}  # document frequency
        self.idf = {}  # inverse document frequency
        
        # BM25 参数
        self.k1 = 1.5  # 可调整
        self.b = 0.75  # 可调整
        
        self._build_index()
    
    def _build_index(self):
        """构建倒排索引"""
        total_length = 0
        
        # 第一遍：分词和计算文档频率
        for doc in self.corpus:
            tokens = self.tokenizer.tokenize_with_ngrams(doc)
            self.tokenized_corpus.append(tokens)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)
            
            # 计算文档频率
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] = self.df.get(token, 0) + 1
        
        # 计算平均文档长度
        self.avg_doc_length = total_length / self.corpus_size if self.corpus_size > 0 else 0
        
        # 计算 IDF
        for token, df in self.df.items():
            # 使用改进的 IDF 公式，避免除零
            self.idf[token] = np.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
    
    def get_scores(self, query: str, boost_exact_match: bool = True) -> np.ndarray:
        """计算查询的 BM25 分数"""
        # 分词查询
        query_tokens = self.tokenizer.tokenize_with_ngrams(query)
        scores = np.zeros(self.corpus_size)
        
        # 计算每个文档的分数
        for idx, doc_tokens in enumerate(self.tokenized_corpus):
            doc_length = self.doc_lengths[idx]
            doc_score = 0.0
            
            # 统计词频
            doc_tf = Counter(doc_tokens)
            
            for token in query_tokens:
                if token not in self.idf:
                    continue
                
                tf = doc_tf.get(token, 0)
                if tf == 0:
                    continue
                
                # BM25 公式
                idf = self.idf[token]
                numerator = idf * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                
                token_score = numerator / denominator
                
                # 提升完全匹配的权重
                if boost_exact_match and token in query.lower():
                    token_score *= 1.2
                
                # 提升短语匹配的权重
                if '_' in token:  # 这是一个 n-gram
                    token_score *= 1.1
                
                doc_score += token_score
            
            scores[idx] = doc_score
        
        return scores




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
        
        self.ngram_config = {
            'use_enhanced_bm25': True,  # 使用增强的 BM25
            'ngram_range': (1, 3),      # 使用 1-gram 到 3-gram
            'boost_exact_match': True,  # 提升完全匹配的权重
            'extract_phrases': True,    # 提取技术短语
        }
        
        # ============ 新增：初始化增强的分词器 ============
        self.enhanced_tokenizer = EnhancedTokenizer()
        self.enhanced_bm25_index = None

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
        
        # Cache paths
        self.idea_pool_path = os.path.join(cache_dir, "idea_pool.json")
        self.embeddings_path = os.path.join(cache_dir, "idea_embeddings.pkl")
        self.bm25_path = os.path.join(cache_dir, "bm25_index.pkl")
        self.enhanced_bm25_path = os.path.join(cache_dir, "enhanced_bm25_index.pkl")  # 新增
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
    
    @lru_cache(maxsize=5000)
    def _tokenize(self, text: str, use_ngrams: bool = False) -> List[str]:
        """Cached tokenization with lowercase and word extraction"""
        if use_ngrams and self.ngram_config['use_enhanced_bm25']:
            # 使用增强的分词器
            return self.enhanced_tokenizer.tokenize_with_ngrams(
                text, 
                n_range=self.ngram_config['ngram_range']
            )
        else:
            # 原有的简单分词
            text = text.lower()
            tokens = re.findall(r'\w+', text)
            return tokens
    
    def _extract_phrases(self, text: str) -> List[str]:
        """提取技术短语"""
        if self.ngram_config['extract_phrases']:
            return self.enhanced_tokenizer.extract_phrases(text)
        return []

    def _expand_query_with_synonyms(self, tokens: List[str]) -> List[str]:
        """Expand query tokens with domain-specific synonyms"""
        expanded_tokens = []
        seen = set()
        
        for token in tokens:
            if token not in seen:
                expanded_tokens.append(token)
                seen.add(token)
            
            # Check for synonyms
            for key, synonyms in self.synonyms.items():
                if token.lower() in [s.lower() for s in synonyms]:
                    for syn in synonyms:
                        syn_tokens = syn.split()
                        for syn_token in syn_tokens:
                            if syn_token.lower() not in seen:
                                expanded_tokens.append(syn_token.lower())
                                seen.add(syn_token.lower())
        
        return expanded_tokens
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract top keywords from text based on frequency"""
        tokens = self._tokenize(text)
        
        # TODO 怎么强化？
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i',
            'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'some', 'any', 'few',
            'more', 'most', 'other', 'such', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 's', 't', 'just', 'don', 'now'
        }
        
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

            if os.path.exists(self.enhanced_bm25_path):
                with builtins.open(self.enhanced_bm25_path, 'rb') as f:
                    self.enhanced_bm25_index = pickle.load(f)
                logger.info("Loaded enhanced BM25 index from cache")
            
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
            
            # Save enhanced BM25 index
            if self.enhanced_bm25_index:
                with builtins.open(self.enhanced_bm25_path, 'wb') as f:
                    pickle.dump(self.enhanced_bm25_index, f)

            # Save FAISS index
            if FAISS_AVAILABLE and self.faiss_index:
                faiss.write_index(self.faiss_index, self.faiss_index_path)
            
            # Save metadata
            metadata = {
                'last_updated': datetime.now().isoformat(),
                'total_ideas': len(self.idea_pool),
                'semantic_weight': self.semantic_weight,
                'keyword_weight': self.keyword_weight,
                'faiss_enabled': FAISS_AVAILABLE and self.faiss_index is not None,
                'enhanced_bm25_enabled': self.ngram_config['use_enhanced_bm25']  # 新增
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
        
        # ============ 新增：重建增强的 BM25 索引 ============
        if self.ngram_config['use_enhanced_bm25']:
            self._rebuild_enhanced_bm25_index()
        
        # Rebuild FAISS index if available
        if FAISS_AVAILABLE:
            self._rebuild_faiss_index()
    
    def _rebuild_enhanced_bm25_index(self):
        """重建增强的 BM25 索引"""
        logger.info("Building enhanced BM25 index with n-grams...")
        
        # 准备语料库
        corpus = []
        for idea_name in self.idea_keys:
            idea_data = self.idea_pool[idea_name]
            text = self._get_idea_text(idea_data)
            corpus.append(text)
        
        # 创建增强的 BM25 索引
        if corpus:
            self.enhanced_bm25_index = EnhancedBM25(corpus, self.enhanced_tokenizer)
            logger.info(f"Built enhanced BM25 index with {len(corpus)} documents")
    
    
    
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
        # 如果启用了增强的 BM25，使用它
        if self.ngram_config['use_enhanced_bm25'] and self.enhanced_bm25_index:
            return self.enhanced_bm25_index.get_scores(
                query, 
                boost_exact_match=self.ngram_config['boost_exact_match']
            )
        
        # 否则使用原有的 BM25
        if not self.bm25_index or not BM25Okapi:
            return np.zeros(len(self.idea_keys))
        
        # Tokenize and expand query
        tokenized_query = self._tokenize(query)
        expanded_query = self._expand_query_with_synonyms(tokenized_query)
        
        scores = self.bm25_index.get_scores(expanded_query)
        
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
        if (not self.bm25_index or not BM25Okapi) and not self.enhanced_bm25_index:
            return np.zeros(len(self.idea_keys))
        
        field_scores = []
        
        # 处理 problem_data 中的字段
        # 映射 problem_data 的字段到 idea_pool 的字段
        field_mapping = {
            'problem': 'problem',
            'reason': 'method',  # 将 reason 映射到 method 的权重
            'label': 'context'   # 将 label 映射到 context 的权重
        }
        # TODO 这里的field_weights包括idea_pool的method,problem,context但是problem_data只有problem,reason,label
        for data_field, idea_field in field_mapping.items():
            if data_field in problem_data and problem_data[data_field]:
                query = problem_data[data_field]
                weight = self.field_weights.get(idea_field, 0.5)
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
        min_score: float = 0.3,
        use_reranking: bool = True
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
        
        # Adjust weights based on query characteristics
        query_length = len(query.split())

        # TODO 是不是有必要改？
        if query_length < 10:
            adj_keyword_weight = min(0.5, self.keyword_weight * 1.5)
        else:
            adj_keyword_weight = max(0.1, self.keyword_weight * 0.7)
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
            if score >= min_score * 0.8:  # Slightly relaxed threshold
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
        
        # Rerank if enabled
        if use_reranking and candidates:
            candidates = self._rerank_candidates(
                query, problem_data, candidates, top_k
            )
        
        # Final selection
        final_results = []
        for candidate in candidates[:top_k]:
            if candidate.get('final_score', candidate['initial_score']) >= min_score:
                # Extract keywords
                idea_text = self._get_idea_text(candidate['idea_data'])
                keywords = self._extract_keywords(idea_text, top_k=5)

                result = {
                    'idea_name': candidate['idea_name'],
                    'similarity_score': float(candidate.get('final_score', candidate['initial_score'])),
                    'semantic_score': float(candidate['semantic_score']),
                    'keyword_score': float(candidate['keyword_score']),
                    'idea_data': candidate['idea_data'],
                    'key_concepts': keywords
                }
                
                if candidate.get('matching_aspects'):
                    result['matching_aspects'] = candidate['matching_aspects']
                
                final_results.append(result)
        
        # Cache results
        self.query_cache[cache_key] = {
            'results': final_results,
            'timestamp': time.time()
        }
        
        logger.info(f"Found {len(final_results)} relevant ideas for problem '{problem_name}'")
        return final_results
    
    def _rerank_candidates(
        self,
        query: str,
        problem_data: Dict[str, str],
        candidates: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Rerank candidates using fine-grained features"""
        problem_keywords = set(self._extract_keywords(
            problem_data.get('problem', ''), 
            15
        ))
        
        # ============ 新增：提取问题中的短语 ============
        problem_phrases = set(self._extract_phrases(problem_data.get('problem', '')))
        
        for candidate in candidates:
            idea_data = candidate['idea_data']
            matching_aspects = []
            bonus_score = 0.0
            
            if isinstance(idea_data, dict):
                # 提取idea的关键词
                idea_keywords = set(self._extract_keywords(
                    self._get_idea_text(idea_data), 15
                ))
                
                # ============ 新增：提取idea的短语 ============
                idea_phrases = set(self._extract_phrases(self._get_idea_text(idea_data)))
                
                # 计算关键词重叠
                overlap = problem_keywords & idea_keywords
                if overlap:
                    overlap_ratio = len(overlap) / max(len(problem_keywords), 1)
                    bonus_score += 0.1 * overlap_ratio
                    matching_aspects.append(f'keyword_overlap_{overlap_ratio:.2f}')
                
                # ============ 新增：计算短语重叠 ============
                phrase_overlap = problem_phrases & idea_phrases
                if phrase_overlap:
                    phrase_overlap_ratio = len(phrase_overlap) / max(len(problem_phrases), 1)
                    bonus_score += 0.15 * phrase_overlap_ratio  # 短语匹配权重更高
                    matching_aspects.append(f'phrase_match_{",".join(list(phrase_overlap)[:3])}')
                
                # 检查是否都提到了相似的技术术语
                problem_text = problem_data.get('problem', '').lower()
                idea_problem = idea_data.get('problem', '').lower()
                
                # 特定技术匹配
                tech_terms = ['efficiency', 'batch', 'padding', 'ensemble', 'cross-validation', 
                             'overfitting', 'variance', 'tokenizer', 'dataloader', 'inference']
                
                matched_terms = []
                for term in tech_terms:
                    if term in problem_text and term in idea_problem:
                        matched_terms.append(term)
                
                if matched_terms:
                    bonus_score += 0.05 * len(matched_terms)
                    matching_aspects.append(f'tech_match_{",".join(matched_terms[:3])}')
                
                # 方法复杂度
                if idea_data.get('method'):
                    method_length = len(idea_data['method'].split())
                    if method_length > 30:  # 详细的方法描述
                        bonus_score += 0.05
                        matching_aspects.append('detailed_method')
                
                # 上下文相似性
                if 'context' in idea_data and idea_data['context']:
                    context_keywords = set(self._extract_keywords(idea_data['context'], 5))
                    context_overlap = problem_keywords & context_keywords
                    if context_overlap:
                        bonus_score += 0.03
                        matching_aspects.append('context_relevant')
            
            # Update final score
            candidate['final_score'] = min(1.0, candidate['initial_score'] + bonus_score)
            candidate['matching_aspects'] = matching_aspects
            candidate['bonus_score'] = bonus_score
        
        # Sort by final score
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        return candidates
    

    def configure_ngram(self, use_enhanced_bm25: bool = True, ngram_range: Tuple[int, int] = (1, 3)):
        """配置 n-gram 设置"""
        self.ngram_config['use_enhanced_bm25'] = use_enhanced_bm25
        self.ngram_config['ngram_range'] = ngram_range
        
        if use_enhanced_bm25:
            # 重建增强的索引
            self._rebuild_enhanced_bm25_index()
        
        logger.info(f"N-gram configuration updated: {self.ngram_config}")

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
        
        for problem_name, problem_data in problems.items():
            section = f"### Problem: {problem_name}\n"
            section += f"**Description**: {problem_data.get('problem', '')}\n"
            if 'reason' in problem_data and problem_data['reason']:
                section += f"**Reason**: {problem_data['reason']}\n"
            if 'label' in problem_data and problem_data['label']:
                section += f"**Label**: {problem_data['label']}\n"
            section += "\n"
            
            # Add relevant ideas as learning references
            relevant_ideas = problem_to_ideas.get(problem_name, [])
            if relevant_ideas:
                section += "**Reference Solutions from Similar Successful Cases**:\n"
                section += "*The following are proven approaches from similar problems that might inspire the hypothesis:*\n\n"
                
                for i, idea_info in enumerate(relevant_ideas[:3], 1):  # Top 3 most relevant
                    idea_name = idea_info['idea_name']
                    score = idea_info['similarity_score']
                    idea_data = idea_info['idea_data']
                    key_concepts = idea_info.get('key_concepts', [])
                    matching_aspects = idea_info.get('matching_aspects', [])
                    
                    section += f"{i}. **{idea_name}** (Relevance: {score:.2f})\n"
                    
                    if isinstance(idea_data, dict):
                        if 'problem' in idea_data:
                            section += f"   - **Similar Problem**: {idea_data['problem']}\n"
                        if 'method' in idea_data:
                            section += f"   - **Successful Method**: {idea_data['method']}\n"
                        if 'context' in idea_data:
                            section += f"   - **Implementation Context**: {idea_data['context']}\n"
                    else:
                        section += f"   - **Description**: {idea_data}\n"
                    
                    if key_concepts:
                        section += f"   - **Key Concepts**: {', '.join(key_concepts[:5])}\n"
                    
                    if matching_aspects:
                        section += f"   - **Matching Aspects**: {', '.join(matching_aspects)}\n"
                    
                    section += "\n"
                
                section += "*Consider adapting these successful approaches to your specific problem context.*\n\n"
            else:
                section += "**Note**: No highly similar cases found in the knowledge base. Consider novel approaches.\n\n"
            
            formatted_sections.append(section)
        
        return "\n".join(formatted_sections)
    
    def hypothesis_draft(
        self,
        no_sota_idea_path: str | None,
        component_desc: str,
        scenario_desc: str,
        exp_and_feedback_list_desc: str,
        sota_exp_desc: str,
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
            scenario_desc, exp_and_feedback_list_desc, formatted_problems
        )
        
        # Enhance hypotheses with references
        enhanced_hypotheses = self._enhance_hypotheses_with_references(
            hypothesis_dict, problem_to_ideas
        )
        
        logger.info(f"Generated {len(enhanced_hypotheses)} hypotheses with enhanced hybrid RAG")
        return enhanced_hypotheses
    
    def _generate_hypotheses_with_ideas(
        self,
        scenario_desc: str,
        exp_and_feedback_list_desc: str,
        formatted_problems: str
    ) -> Dict[str, Dict[str, Any]]:
        """Generate hypotheses using the formatted problems and ideas"""
        try:
            # Import necessary components

            
            # Prepare prompts using template system
            sys_prompt = T(".prompts_v2:hypothesis_draft.system").r(
                hypothesis_spec=T(".prompts_v2:specification.hypothesis").r(),
                hypothesis_output_format=T(".prompts_v2:output_format.hypothesis").r(),
            )
            
            user_prompt = T(".prompts_v2:hypothesis_draft.user").r(
                scenario_desc=scenario_desc,
                exp_and_feedback_list_desc=exp_and_feedback_list_desc,
                problems=formatted_problems,
                enable_idea_pool=True,
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
    
    def _enhance_hypotheses_with_references(
        self,
        hypothesis_dict: Dict[str, Dict],
        problem_to_ideas: Dict[str, List[Dict]]
    ) -> Dict[str, Dict]:
        """Enhance generated hypotheses with references to relevant ideas"""
        enhanced_hypotheses = {}
        
        for problem_name, hypothesis in hypothesis_dict.items():
            # Copy original hypothesis
            enhanced_hypothesis = hypothesis.copy()
            
            # Add references to relevant ideas
            relevant_ideas = problem_to_ideas.get(problem_name, [])
            if relevant_ideas:
                # Add top 3 supporting ideas
                supporting_refs = []
                for idea_info in relevant_ideas[:3]:
                    ref_info = {
                        'idea_name': idea_info['idea_name'],
                        'relevance_score': round(idea_info['similarity_score'], 3),
                        'key_concepts': idea_info.get('key_concepts', [])[:5]
                    }
                    
                    # Add matching aspects if available
                    if idea_info.get('matching_aspects'):
                        ref_info['matching_aspects'] = idea_info['matching_aspects']
                    
                    # Add brief method summary
                    if isinstance(idea_info['idea_data'], dict) and 'method' in idea_info['idea_data']:
                        method_preview = idea_info['idea_data']['method'][:100] + "..."
                        ref_info['method_preview'] = method_preview
                    
                    supporting_refs.append(ref_info)
                
                enhanced_hypothesis['supporting_ideas'] = supporting_refs
                enhanced_hypothesis['evidence_based'] = True
                
                # Set confidence level based on top match score
                top_score = relevant_ideas[0]['similarity_score']
                if top_score > 0.8:
                    enhanced_hypothesis['confidence_level'] = 'very_high'
                elif top_score > 0.6:
                    enhanced_hypothesis['confidence_level'] = 'high'
                elif top_score > 0.4:
                    enhanced_hypothesis['confidence_level'] = 'medium'
                else:
                    enhanced_hypothesis['confidence_level'] = 'low'
                    
                # Add retrieval metadata
                enhanced_hypothesis['retrieval_metadata'] = {
                    'num_relevant_ideas': len(relevant_ideas),
                    'avg_relevance_score': round(
                        sum(idea['similarity_score'] for idea in relevant_ideas) / len(relevant_ideas), 
                        3
                    ),
                    'retrieval_timestamp': datetime.now().isoformat()
                }
            else:
                enhanced_hypothesis['evidence_based'] = False
                enhanced_hypothesis['confidence_level'] = 'exploratory'
                enhanced_hypothesis['note'] = 'No similar cases found - novel hypothesis'
            
            enhanced_hypotheses[problem_name] = enhanced_hypothesis
        
        return enhanced_hypotheses
    
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