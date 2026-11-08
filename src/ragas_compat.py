"""Compatibility shim, imported before ragas anywhere in this repo.

ragas 0.4.3 imports ChatVertexAI from
langchain_community.chat_models.vertexai, a module that
langchain-community 0.4 removed when the class moved to the
langchain-google-vertexai package. The class ragas wants is
the same class, now imported from its new home and registered
under the old module path. ragas uses it only in an isinstance
dispatch list; this lab never routes through Vertex AI. The
shim is recorded as DECISIONS.md D2.
"""
import sys
import types


def apply() -> None:
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
        return
    except ImportError:
        pass
    from langchain_google_vertexai import ChatVertexAI

    module = types.ModuleType(name)
    module.ChatVertexAI = ChatVertexAI
    sys.modules[name] = module
