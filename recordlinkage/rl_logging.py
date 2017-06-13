"""Logging utilities."""

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# Modifications copyright Jonathan de Bruin 2017

# pylint: disable=unused-import

import logging as _logging
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN
import sys as _sys

# Determine whether we are in an interactive environment
_interactive = False
try:
    # This is only defined in interactive shells
    if _sys.ps1:
        _interactive = True
except AttributeError:
    # Even now, we may be in an interactive shell with `python -i`.
    _interactive = _sys.flags.interactive

# Scope the tensorflow logger to not conflict with users' loggers
_logger = _logging.getLogger('recordlinkage')

# If we are in an interactive environment (like jupyter), set loglevel to info
# and pipe the output to stdout
if _interactive:
    _logger.setLevel(WARN)
    _logging_target = _sys.stdout
else:
    _logging_target = _sys.stderr

# Add the output handler
_handler = _logging.StreamHandler(_logging_target)
_handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
_logger.addHandler(_handler)

log = _logger.log
debug = _logger.debug
error = _logger.error
fatal = _logger.fatal
info = _logger.info
warn = _logger.warn
warning = _logger.warning


def get_verbosity():
    """Return how much logging output will be produced."""
    return _logger.getEffectiveLevel()


def set_verbosity(verbosity):
    """Sets the threshold for what messages will be logged."""
    _logger.setLevel(verbosity)
