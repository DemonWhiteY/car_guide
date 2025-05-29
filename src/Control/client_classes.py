# client_classes.py (各模块客户端类)
from base_client import BaseClient


# UI 客户端
class UICommunication:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = BaseClient('ui')
        return cls._instance


# LLM 客户端
class LLMCommunication:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = BaseClient('llm')
        return cls._instance


# Face 客户端
class FaceCommunication:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = BaseClient('face')
        return cls._instance


# Gesture 客户端
class GestureCommunication:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = BaseClient('gesture')
        return cls._instance


# Voice 客户端
class VoiceCommunication:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = BaseClient('voice')
        return cls._instance


