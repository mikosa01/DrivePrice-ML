
class BaseError(Exception):
    """Base Model Error"""


class CustomInvalidModelInput(BaseError):
    """Invalid Model Input"""
    pass

class CustomValueError(BaseError):
    """Custom Value Error Input"""
    pass

class CustomFileNotFoundError(BaseError):
    """Custom File Not Found Error"""
    pass