
class BaseError(Exception):
    """Base Model Error"""


class CustomInvalidModelInput(BaseError):
    """Invalid Model Input"""
    pass

class CustomValueError(BaseError):
    """Custom Value Error Input"""
    pass