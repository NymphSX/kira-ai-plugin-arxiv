# kira-ai-plugin-arxiv
try:
    from .main import ArxivPlugin
    __all__ = ["ArxivPlugin"]
except ImportError:
    # Dependency not installed (e.g. in test environment)
    __all__ = []
