import sys  # System utilities for exception traceback extraction
from src.logger import logging  # Centralized logging for error tracking


def error_message_detail(error_message: str, error_detail: sys) -> str:
    """
    Extracts precise error location (file + line) from traceback and formats 
    comprehensive error message for debugging ML pipelines.
    
    Args:
        error_message (str): User-provided error description
        error_detail: sys.exc_info() object containing full traceback context
        
    Returns:
        str: Complete error string with filename, line number, and message
    """
    # Unpack traceback info: (exception_type, exception_value, traceback_object)
    _, _, exc_tb = error_detail.exc_info()
    
    # Extract exact filename where error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Build detailed error context for production debugging
    formatted_error = (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] "
        f"error message[{str(error_message)}]"
    )
    
    return formatted_error


class CustomException(Exception):
    """
    Production-ready custom exception that automatically:
    1. Captures exact error location (file + line number)
    2. Logs full error context to centralized log file
    3. Provides clean string representation for print/raise
    """
    
    def __init__(self, error_message: str, error_detail: sys):
        """
        Initialize with error message and traceback, auto-logging the full context.
        
        Args:
            error_message (str): Descriptive error message
            error_detail: sys.exc_info() for location tracking
        """
        # Store original message in base Exception
        super().__init__(error_message)
        
        # Generate detailed error with location context
        self.error_message = error_message_detail(
            error_message=error_message,
            error_detail=error_detail
        )
        
        # Automatically log full error context (timestamped to ./logs/)
        logging.error(self.error_message)
    
    def __str__(self) -> str:
        """Clean string representation for exception printing."""
        return self.error_message
