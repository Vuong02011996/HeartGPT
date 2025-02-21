import traceback
from .custom_logging import LOGGING


LOGGING_SESSION = LOGGING()


def write_error_log(
        error,
        **kwargs
) -> None:
    
    exc = traceback.sys.exc_info()
    msg = str(error)
    if exc[2] is not None:
        _:    str = f'{kwargs["class_name"]}.' if 'class_name' in kwargs.keys() else ''
        __:   str = f'[{kwargs["addition"]}] ' if 'addition' in kwargs.keys() else ''
        
        msg = f'++ {__}[{_}{exc[2].tb_frame.f_code.co_name} error at line {exc[-1].tb_lineno}] - {error}'

    LOGGING_SESSION.error(message=msg)


def get_error_exception(
        error,
        **kwargs
) -> None:
    exc = traceback.sys.exc_info()
    msg = str(error)
    if exc[2] is not None:
        _: str = f'{kwargs["class_name"]}.' if 'class_name' in kwargs.keys() else ''
        __: str = f'[{kwargs["addition"]}] ' if 'addition' in kwargs.keys() else ''
        
        msg = f'++ {__}[{_}{exc[2].tb_frame.f_code.co_name} error at line {exc[-1].tb_lineno}] - {error}'

    raise Exception(msg)
