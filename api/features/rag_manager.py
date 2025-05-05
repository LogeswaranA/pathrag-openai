"""
PathRAG Manager - Centralized management of PathRAG instances
"""

import os
import logging
from typing import override
from PathRAG import PathRAG
from PathRAG.llmv1 import gpt_4o_mini_complete
from dotenv import load_dotenv
load_dotenv(override=True)
logger = logging.getLogger("PathRAG")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
base_url="https://api.openai.com/v1"
os.environ["OPENAI_API_BASE"]=base_url

# Setup a working directory for PathRAG.
WORKING_DIR = os.path.join(os.getcwd(), 'data')
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Global PathRAG instance
_rag_instance = None

def get_rag_instance():
    """
    Get the current PathRAG instance, initializing it if necessary.
    """
    global _rag_instance
    
    if _rag_instance is None:
        logger.info("Initializing PathRAG instance...")
        _rag_instance = PathRAG(
            working_dir=WORKING_DIR,
            llm_model_func=gpt_4o_mini_complete,
        )
        logger.info("PathRAG instance initialized successfully")
    
    return _rag_instance

def reload_rag_instance():
    """
    Reload the PathRAG instance to recognize new data files.
    """
    global _rag_instance
    
    logger.info("Reloading PathRAG instance...")
    
    # Create a new instance
    _rag_instance = PathRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
    )
    
    logger.info("PathRAG instance reloaded successfully")
    return _rag_instance

# Initialize the instance on module import
rag = get_rag_instance()
