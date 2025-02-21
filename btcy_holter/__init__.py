# # -------------------------------------------------
# region Importing the standard library
import os
import gc
import re
import sys
import json
import math
import time
import grpc
import glob
import boto3
import scipy
import signal
import struct
import shutil
import threading
import operator
import datetime
import traceback
import functools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import polars as pl
import wfdb as wf
# endregion Importing the standard library


# # -------------------------------------------------
# region From the standard library
from tqdm import (
    tqdm
)

from pathlib import (
    Path
)

from copy import (
    deepcopy
)

from operator import (
    is_not
)

from collections import (
    Counter
)

from multiprocessing import (
    Pool
)

from bson.objectid import (
    ObjectId
)

from statistics import (
    geometric_mean
)

from enum import (
    Enum
)

from prettytable import (
    PrettyTable
)

from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed
)

from datetime import (
    datetime,
    timedelta
)

from wfdb.processing import (
    resample_sig
)

from functools import (
    partial,
    reduce
)

from typing import (
    Union,
    Dict,
    Final,
    List,
    Tuple,
    Any,
)

from numpy.typing import (
    NDArray,
    ArrayLike,
    DTypeLike
)

from abc import (
    ABC,
    ABCMeta,
    abstractmethod
)

from os.path import (
    basename,
    exists,
    dirname,
    isfile,
    isdir,
    abspath,
    join
)

from itertools import (
    repeat,
    chain,
    starmap
)

from importlib.machinery import (
    SourceFileLoader
)

from grpc_tools import (
    protoc
)
# endregion From the standard library


# # -------------------------------------------------
# region Ignore Warnings
import warnings

for category in [
        UserWarning,
        np.exceptions.VisibleDeprecationWarning
]:
    try:
        warnings.filterwarnings(
                action="ignore",
                category=category
        )
        
    except (Exception,) as except_err:
        pass

for category in [
    FutureWarning,
    RuntimeWarning,
    DeprecationWarning,
    pl.MapWithoutReturnDtypeWarning
]:
    try:
        warnings.simplefilter(
                action="ignore",
                category=category
        )
    
    except (Exception,) as except_err:
        pass
    
# endregion Ignore Warnings


# # -------------------------------------------------
# region Importing Matplotlib
try:
    import matplotlib
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (19.2, 10.08)
    plt.rcParams.update({'font.size': 8})
    matplotlib.use('TkAgg')

except (Exception,) as except_err:
    pass
# endregion Importing Matplotlib


# # -------------------------------------------------
# region Importing Tensorflow
try:
    import tensorflow as tf

    from tensorflow_serving.apis import (
        predict_pb2,
        prediction_service_pb2_grpc
    )

except (Exception,) as except_err:
    pass
# endregion Importing Tensorflow


# # -------------------------------------------------
# region Importing from the third-party library

sys.path.append(dirname(abspath(__file__)))

from btcy_holter.version import      (__name__, __version__)
from btcy_holter import structs      as sr
from btcy_holter import config       as cf
from btcy_holter import stream       as st
from btcy_holter import define       as df
from btcy_holter import helpers      as hl
from btcy_holter import utils        as ut
from btcy_holter import patterns     as pt
from btcy_holter import sqs          as sq
from btcy_holter import calculate    as cl
from btcy_holter import algs         as al
from btcy_holter import ai_core      as cr
from btcy_holter import report       as rp

# endregion Importing from the third-party library
