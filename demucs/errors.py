# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Exception classes for the Demucs package.
"""


class DemucsError(Exception):
    """
    Base exception class for all Demucs-specific errors.
    """

    pass


class LoadAudioError(DemucsError):
    """
    Exception raised when audio loading fails.
    """

    pass


class ModelLoadingError(DemucsError):
    """
    Exception raised when model loading fails.
    """

    pass


class SegmentValidationError(DemucsError):
    """
    Exception raised when segment parameter is invalid for the model.
    """

    pass
