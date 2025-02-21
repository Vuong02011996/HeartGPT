import logging
import colorlog

from typing import Final
from pytz import timezone, utc
from datetime import datetime

from btcy_holter import cf


LOGGER = {}
LOGGING_TZ = timezone("Asia/Ho_Chi_Minh")


def custom_utc(*args):
    utc_dt = utc.localize(datetime.utcnow())
    converted = utc_dt.astimezone(LOGGING_TZ)

    return converted.timetuple()


class LOGGING:
    ENABLE_LOGGING:    Final[bool] = cf.ENABLE_LOGGING
    
    # Define logging levels
    VERBOSE = 15
    logging.addLevelName(VERBOSE, 'VERBOSE')

    TIMEIT = 60
    logging.addLevelName(TIMEIT, 'TIMEIT')

    # Define log color map
    log_color = {
        'DEBUG':        'cyan',
        'INFO':         'white',
        'WARNING':      'yellow',
        'ERROR':        'red',
        'CRITICAL':     'red,bg_white',
        'VERBOSE':      'cyan',
        'TIMEIT':       'green',
    }

    # Initialize logger
    if LOGGER.get('ai'):
        logger = LOGGER['ai']
    else:
        handler = colorlog.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            fmt='%(log_color)s[%(asctime)s] %(message)s',
            datefmt=f'%d/%m/%y %H:%M:%S {LOGGING_TZ.zone}',
            log_colors=log_color,
            secondary_log_colors={},
            style='%',
            reset=True,
        )
        
        c_handler = logging.StreamHandler()
        c_handler.setFormatter(formatter)
        logger = logging.getLogger('ai')
        logger.handlers.clear()
        if not len(logger.handlers):
            logger.setLevel(logging.DEBUG)
            logger.addHandler(c_handler)
            logger.propagate = False
            
            logging.Formatter.converter = custom_utc
            
        LOGGER['ai'] = logger
    
    @classmethod
    def verbose(
            cls,
            message,
            *args,
            **kwargs
    ) -> None:
        cls.ENABLE_LOGGING and cls.logger.log(cls.VERBOSE, message, *args, **kwargs)

    @classmethod
    def debug(
            cls,
            message,
            *args,
            **kwargs
    ) -> None:
        cls.ENABLE_LOGGING and cls.logger.debug(cls.__generate_message(message), *args, **kwargs)

    @classmethod
    def info(
            cls,
            message,
            *args,
            **kwargs
    ) -> None:
        cls.ENABLE_LOGGING and cls.logger.info(cls.__generate_message(message), *args, **kwargs)

    @classmethod
    def warning(
            cls,
            message,
            *args,
            **kwargs
    ) -> None:
        cls.ENABLE_LOGGING and cls.logger.warning(cls.__generate_message(message), *args, **kwargs)

    @classmethod
    def error(
            cls,
            message,
            *args,
            **kwargs
    ):
        cls.ENABLE_LOGGING and cls.logger.error(cls.__generate_message(message), *args, **kwargs)

    @classmethod
    def critical(
            cls,
            message,
            *args,
            **kwargs
    ) -> None:
        cls.ENABLE_LOGGING and cls.logger.critical(cls.__generate_message(message), *args, **kwargs)

    @classmethod
    def timeit(
            cls,
            message,
            *args,
            **kwargs
    ) -> None:
        msg = cls.__generate_message(message)
        cls.ENABLE_LOGGING and cls.logger.log(cls.TIMEIT, msg, *args, **kwargs)

    @staticmethod
    def __generate_message(
            message
    ) -> str:
        additions = ''
        if cf.GLOBAL_STUDY_ID is not None:
            additions += f'[{cf.GLOBAL_STUDY_ID}] '
        
        if cf.GLOBAL_MESSAGE_ID is not None:
            additions += f'[{cf.GLOBAL_MESSAGE_ID}] '

        message = f'{additions}{message}'

        return message
