import numpy as np
import os
import time
from typing import List, Optional
import random
from enum import Enum
from dataclasses import dataclass, field

# 导入必要的库
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None


class EmbeddingType(Enum):
    SENTENCE_TRANSFORMER = "sentence_transformer"
    TFIDF = "tfidf"
    BOW = "bag_of_words"
    HYBRID = "hybrid"


@dataclass
class EmbeddingConfig:
    model_name: str = 'all-MiniLM-L6-v2'
    max_features: int = 1000
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    stop_words: Optional[str] = 'english'
    fallback_dim: int = 100


class LocalEmbeddingManager:
    """本地嵌入管理器，提供多种文本嵌入方案"""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.embedding_model = None
        self.tfidf_vectorizer = None
        self.embedding_type = EmbeddingType.BOW  # 默认回退类型
        self.is_tfidf_fitted = False  # 新增：TF-IDF拟合状态
        self._initialize_embedding_model()
        self._initialize_tfidf_model()

    def _initialize_embedding_model(self):
        """初始化SentenceTransformer嵌入模型"""
        if SentenceTransformer is None:
            print("SentenceTransformer库未安装，将使用TF-IDF或词袋模型作为替代")
            return

        try:
            print(f"正在加载SentenceTransformer嵌入模型: {self.config.model_name}...")
            self.embedding_model = SentenceTransformer(self.config.model_name)
            self.embedding_type = EmbeddingType.SENTENCE_TRANSFORMER
            print("嵌入模型加载成功！")
        except Exception as e:
            print(f"嵌入模型加载失败: {e}")
            self.embedding_model = None

    def _initialize_tfidf_model(self):
        """初始化TF-IDF模型作为备用方案"""
        if TfidfVectorizer is None:
            print("sklearn库未安装，无法使用TF-IDF模型")
            return

        try:
            print("正在初始化TF-IDF Vectorizer...")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                stop_words=self.config.stop_words,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df
            )
            self.is_tfidf_fitted = False  # 初始化为未拟合
            print("TF-IDF Vectorizer初始化成功！")
        except Exception as e:
            print(f"TF-IDF模型初始化失败: {e}")
            self.tfidf_vectorizer = None

    def fit_tfidf_vectorizer(self, texts: List[str]):
        """使用历史任务文本训练TF-IDF模型"""
        if self.tfidf_vectorizer is None or not texts:
            return False

        try:
            print(f"使用 {len(texts)} 个历史任务训练TF-IDF模型...")
            self.tfidf_vectorizer.fit(texts)
            self.is_tfidf_fitted = True
            print("TF-IDF模型训练完成！")
            return True
        except Exception as e:
            print(f"TF-IDF模型训练失败: {e}")
            self.is_tfidf_fitted = False
            return False

    def is_tfidf_ready(self) -> bool:
        """检查TF-IDF是否已准备好使用"""
        return self.tfidf_vectorizer is not None and self.is_tfidf_fitted

    def compute_embeddings(self, text: str) -> np.ndarray:
        """计算文本嵌入向量"""
        # 如果所有高级模型都不可用，回退到简易词袋模型
        if self.embedding_model is None and self.tfidf_vectorizer is None:
            return self._compute_bow_embedding(text)

        # 优先尝试使用SentenceTransformer
        if self.embedding_model is not None:
            try:
                embeddings = self.embedding_model.encode([text])[0]
                return np.array(embeddings)
            except Exception as e:
                print(f"SentenceTransformer生成错误，回退到TF-IDF: {e}")

        # 尝试使用TF-IDF
        if self.tfidf_vectorizer is not None:
            try:
                tfidf_vector = self.tfidf_vectorizer.transform([text])
                return tfidf_vector.toarray()[0]
            except Exception as e:
                print(f"TF-IDF生成错误，回退到词袋模型: {e}")

        # 最终回退到简易词袋模型
        return self._compute_bow_embedding(text)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度 - 健壮实现"""
        # 1. 首选：SentenceTransformer（如果可用）
        if self.embedding_model is not None:
            try:
                embeddings = self.embedding_model.encode([text1, text2])
                sim = np.dot(embeddings[0], embeddings[1]) / (
                        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
                return float(sim)
            except Exception as e:
                print(f"SentenceTransformer计算相似度失败: {e}")

        # 2. 次选：TF-IDF（如果已拟合）
        if self.is_tfidf_ready():
            try:
                tfidf_matrix = self.tfidf_vectorizer.transform([text1, text2])
                if cosine_similarity is not None:
                    sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
                    return float(sim)
            except Exception as e:
                print(f"TF-IDF计算相似度失败: {e}")

        # 3. 回退：简易文本相似度
        return self._compute_simple_similarity(text1, text2)

    def _compute_bow_embedding(self, text: str) -> np.ndarray:
        """计算简易词袋模型嵌入"""
        print("使用简易词袋模型作为嵌入回退方案")
        words = set(text.lower().split())
        vector = np.zeros(self.config.fallback_dim)
        for i, word in enumerate(words):
            vector[hash(word) % self.config.fallback_dim] = 1
        return vector

    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度"""
        if self.embedding_model is not None:
            try:
                embeddings = self.embedding_model.encode([text1, text2])
                sim = np.dot(embeddings[0], embeddings[1]) / (
                            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
                return float(sim)
            except Exception as e:
                print(f"使用SentenceTransformer计算相似度失败: {e}")

        if self.tfidf_vectorizer is not None and hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            try:
                tfidf_matrix = self.tfidf_vectorizer.transform([text1, text2])
                if cosine_similarity is not None:
                    sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
                    return float(sim)
            except Exception as e:
                print(f"使用TF-IDF计算相似度失败: {e}")

        # 使用简易相似度计算
        return self._compute_simple_similarity(text1, text2)

    def _compute_simple_similarity(self, text1: str, text2: str) -> float:
        """使用简单方法计算文本相似度"""
        import difflib
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def get_embedding_type(self) -> str:
        """获取当前使用的嵌入类型"""
        if self.embedding_model is not None:
            return EmbeddingType.SENTENCE_TRANSFORMER.value
        elif self.tfidf_vectorizer is not None:
            return EmbeddingType.TFIDF.value
        else:
            return EmbeddingType.BOW.value

    def get_embedding_dim(self) -> int:
        """获取嵌入向量维度"""
        if self.embedding_model is not None:
            try:
                sample = self.embedding_model.encode(["test"])[0]
                return len(sample)
            except:
                pass
        if self.tfidf_vectorizer is not None and hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            return len(self.tfidf_vectorizer.vocabulary_)
        return self.config.fallback_dim