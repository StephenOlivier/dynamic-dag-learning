"""
Models 模块
===========
提供LLM和嵌入模型的管理功能，支持多种模型和回退机制。

主要组件:
- LLMManager: 统一的LLM接口管理器
- LocalEmbeddingManager: 本地嵌入模型管理器
- EmbeddingConfig: 嵌入模型配置
- EmbeddingType: 嵌入类型枚举

使用示例:
    from models import LLMManager, LocalEmbeddingManager

    llm_manager = LLMManager()
    response = llm_manager.generate_response("Hello, how are you?")

    embedding_manager = LocalEmbeddingManager()
    embeddings = embedding_manager.compute_embeddings("Sample text")
"""