"""Exceptions for the video-to-reference extraction feature.

Every failure mode that can reach an API caller is a distinct, typed
exception here so route handlers can map it to a clean HTTP status code
instead of leaking a stack trace or a local filesystem path.
"""

from __future__ import annotations


class VideoExtractionError(Exception):
    """Base class for all video-extraction errors."""


class InvalidVideoError(VideoExtractionError):
    """The uploaded file is not a readable/valid supported video."""


class UnsupportedFormatError(InvalidVideoError):
    """File extension, container or codec is not supported."""


class VideoTooLargeError(InvalidVideoError):
    """Upload exceeds the configured size limit."""


class VideoDurationError(InvalidVideoError):
    """Video duration is outside the configured bounds."""


class VideoResolutionError(InvalidVideoError):
    """Video resolution or frame rate is outside the configured bounds."""


class JobNotFoundError(VideoExtractionError):
    """No job exists with the given id (for this store)."""


class JobOwnershipError(VideoExtractionError):
    """The requesting user does not own this job and is not an admin."""


class JobStateError(VideoExtractionError):
    """The requested operation is not valid for the job's current status."""


class ProcessingError(VideoExtractionError):
    """Something went wrong while extracting frames/measurements."""


class InsufficientMeasurementsError(VideoExtractionError):
    """Not enough approved views/measurements to run the deformation pipeline."""


class StorageError(VideoExtractionError):
    """Local filesystem storage backend failed to read/write/delete a file."""
