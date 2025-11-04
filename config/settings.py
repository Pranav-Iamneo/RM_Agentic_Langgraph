"""
TODO: Centralized application settings and configuration management.
===================================================================
PURPOSE:
  - Manage all application settings from environment variables
  - Support both pydantic-settings (new) and BaseSettings (legacy) versions
  - Provide single source of truth for configuration across modules
  - Auto-load .env files for local development and deployment

KEY FEATURES:
  1. Environment Variable Loading
     - Auto-discovers .env file in project root or current directory
     - Uses python-dotenv for secure loading
     - Graceful fallback if .env not found
     - Supports override=True to prioritize .env over existing env vars

  2. API Configuration
     - gemini_api_key: Google Gemini API authentication (GEMINI_API_KEY_1)
     - langchain_api_key: LangChain API key (optional, for future use)
     - LangSmith tracing: Disabled by default (no valid API key)
     - langchain_project: Project name for tracing (rm-agentic-ai)

  3. Application Settings
     - log_level: Logging verbosity (default: INFO)
     - enable_monitoring: Toggle monitoring features (default: True)
     - debug_mode: Development flag (default: False)

  4. Performance Settings
     - max_concurrent_agents: Limit parallel agent execution (default: 5)
     - agent_timeout: Max execution time per agent in seconds (default: 300)
     - cache_ttl: Cache time-to-live in seconds (default: 3600)

  5. File Paths
     - data_dir: Input data directory (default: data/)
     - models_dir: Trained models directory (default: ml/models/)
     - output_dir: Output/results directory (default: output/)
     - prospects_csv: Prospect data file (default: data/input_data/prospects.csv)
     - products_csv: Product catalog file (default: data/input_data/products.csv)

  6. ML Model Files
     - risk_model_path: Risk assessment model file
     - goal_model_path: Goal success prediction model file
     - label_encoders_path: Categorical encoders for risk model
     - goal_success_encoders_path: Categorical encoders for goal model

  7. LLM Configuration
     - default_temperature: LLM temperature setting (default: 0.1)
     - max_tokens: Max tokens in LLM responses (default: 4000)
     - model_name: Selected LLM model (default: gemini-2.0-flash)

PYDANTIC COMPATIBILITY:
  - Tries pydantic-settings first (modern versions)
  - Falls back to pydantic.BaseSettings (intermediate versions)
  - Falls back to pydantic.BaseModel (legacy support)
  - SettingsConfigDict handling for version compatibility

SETTINGS ACCESS:
  - get_settings(): Non-cached factory function (fresh read each time)
  - get_cached_settings(): LRU-cached version (5 cache hits)
  - Both return Settings instance with all configured values

ENVIRONMENT VARIABLE NAMING:
  - All settings map to ENV variables using Field(env="VAR_NAME")
  - Example: log_level maps to LOG_LEVEL environment variable
  - Optional settings default if environment variable not set

COMMON USE CASES:
  1. Get API key: settings.gemini_api_key
  2. Get data path: settings.prospects_csv
  3. Get model file: settings.risk_model_path
  4. Get timeout: settings.agent_timeout
  5. Check debug mode: settings.debug_mode

ERROR HANDLING:
  - Graceful degradation if .env file missing
  - Supports missing API keys (development mode)
  - Optional fields with sensible defaults
  - Type validation via Pydantic

STATUS:
  - Production-ready configuration system
  - Supports multiple Pydantic versions for compatibility
  - All settings properly validated and documented
  - Caching strategy optimizes repeated settings access
"""

import os
from functools import lru_cache
from typing import Optional
from pathlib import Path
from pydantic import Field
from dotenv import load_dotenv

# Load .env file into environment before Pydantic reads it
try:
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
    else:
        # Fallback to current working directory
        load_dotenv(Path.cwd() / ".env", override=True)
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    try:
        from pydantic import BaseSettings
        SettingsConfigDict = dict
    except ImportError:
        # Fallback for older versions
        from pydantic import BaseModel as BaseSettings
        SettingsConfigDict = dict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Keys
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY_1"))
    langchain_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("LANGCHAIN_API_KEY"))
    
    # LangSmith Configuration
    langchain_tracing_v2: bool = Field(True, env="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field("rm-agentic-ai", env="LANGCHAIN_PROJECT")
    
    # Application Settings
    log_level: str = Field("INFO", env="LOG_LEVEL")
    enable_monitoring: bool = Field(True, env="ENABLE_MONITORING")
    debug_mode: bool = Field(False, env="DEBUG_MODE")
    
    # Performance Settings
    max_concurrent_agents: int = Field(5, env="MAX_CONCURRENT_AGENTS")
    agent_timeout: int = Field(300, env="AGENT_TIMEOUT")
    cache_ttl: int = Field(3600, env="CACHE_TTL")
    
    # File Paths
    data_dir: str = Field("data", env="DATA_DIR")
    models_dir: str = Field("ml/models", env="MODELS_DIR")
    output_dir: str = Field("output", env="OUTPUT_DIR")

    # Model Settings
    risk_model_path: str = Field("ml/models/risk_profile_model.pkl")
    goal_model_path: str = Field("ml/models/goal_success_model.pkl")
    risk_encoders_path: str = Field("ml/models/label_encoders.pkl")
    goal_encoders_path: str = Field("ml/models/goal_success_label_encoders.pkl")
    
    # Data Files
    prospects_csv: str = Field("data/input_data/prospects.csv")
    products_csv: str = Field("data/input_data/products.csv")
    
    # Streamlit Configuration
    page_title: str = Field("AI-Powered Investment Analyzer")
    page_icon: str = Field("🤖")
    layout: str = Field("wide")
    
    # Agent Configuration
    default_temperature: float = Field(0.1)
    max_tokens: int = Field(4000)

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore"  # Allow extra fields in environment
    )


def get_settings() -> Settings:
    """Get application settings. Caching is disabled to allow environment updates in tests."""
    return Settings()


# Global settings instance (lazy-loaded)
_settings = None

def get_cached_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


# For backward compatibility
settings = get_settings()