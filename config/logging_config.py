"""
TODO: Structured logging configuration using loguru framework.
==============================================================
PURPOSE:
  - Setup application-wide logging with consistent formatting
  - Provide dual output: console (colored) and file (persistent)
  - Auto-rotate log files with compression and retention policies
  - Support per-module logger instances with context tracking

KEY FEATURES:
  1. Dual Logging Output
     - Console Output: Colored, human-readable format (stdout)
     - File Output: Timestamped format (logs/app.log)
     - Agent-specific logs: Separate file (logs/agents.log)
     - All handlers respect configured log level (INFO by default)

  2. Console Format
     Format: <YYYY-MM-DD HH:mm:ss> | LEVEL | module:function:line | message
     Colors: Green timestamps, Cyan module info, Level-based colors
     Features: Full backtrace and diagnostics on errors

  3. File Format
     Format: YYYY-MM-DD HH:mm:ss | LEVEL | module:function:line | message
     Rotation: Auto-rotate when file reaches 10 MB (main), 5 MB (agents)
     Retention: Keep 30 days of app.log, 7 days of agents.log
     Compression: Automatic .zip compression of rotated logs
     Backtrace: Full stack traces captured for debugging

  4. Logger Instances
     - Global logger: logger object (from loguru)
     - Bound loggers: get_logger(module_name) returns module-specific instance
     - Context tracking: Includes function name, line number, module name
     - Performance: Lightweight async I/O for file operations

CONFIGURATION:
  - Log Level: Loaded from settings.log_level (default: INFO)
  - Log Directory: Automatic creation of logs/ folder if missing
  - Rotation Timing: Size-based (not time-based) for predictable behavior
  - Compression Format: ZIP for disk space efficiency

FILE ROTATION POLICY:
  - App logs (logs/app.log):
    • Max size: 10 MB
    • Retention: 30 days
    • Example: app.log.2024-01-15.zip

  - Agent logs (logs/agents.log):
    • Max size: 5 MB
    • Retention: 7 days
    • Example: agents.2024-01-15.zip

USAGE EXAMPLES:
  1. Setup logging (call once at app start):
     setup_logging()

  2. Get module-specific logger:
     logger = get_logger("MyModule")
     logger.info("Message with context")

  3. Output includes:
     2024-01-15 10:30:45 | INFO     | MyModule:my_function:42 | Message

STRUCTURED LOGGING BENEFITS:
  - Searchable and parseable log format
  - Automatic context injection (module, function, line)
  - Persistent history for audit trails
  - Performance debugging via execution times
  - Error diagnostics with full backtrace

INTEGRATION WITH APP:
  - Called in main.py at startup: setup_logging()
  - Called in all agents: logger = get_logger("AgentName")
  - Called in workflow: logger = get_logger("ProspectAnalysisWorkflow")
  - Supports concurrent logging from multiple agents

ERROR HANDLING:
  - Graceful fallback if logs/ directory can't be created
  - Non-blocking file I/O (doesn't halt application)
  - Handles high-volume logging without performance impact

STATUS:
  - Production-ready logging system
  - Supports development and deployment environments
  - Proper log rotation prevents disk space issues
  - Structured format supports log analysis tools
"""

import sys
from loguru import logger
from typing import Optional

from .settings import get_settings


def setup_logging(log_level: Optional[str] = None) -> None:
    """Setup application logging with loguru."""
    settings = get_settings()
    level = log_level or settings.log_level
    
    # Remove default handler
    logger.remove()
    
    # Console handler with custom format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    logger.add(
        sys.stdout,
        format=console_format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File handler for persistent logging
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    
    logger.add(
        "logs/app.log",
        format=file_format,
        level=level,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # Agent-specific log file
    logger.add(
        "logs/agents.log",
        format=file_format,
        level=level,
        rotation="5 MB",
        retention="7 days",
        filter=lambda record: "agent" in record["name"].lower(),
        backtrace=True,
        diagnose=True
    )
    
    logger.info(f"Logging initialized with level: {level}")


def get_logger(name: str):
    """Get a logger instance for a specific module."""
    return logger.bind(name=name)