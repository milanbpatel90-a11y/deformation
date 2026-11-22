class VideoValidationError(ValueError):
    """Uploaded video failed validation."""

class VideoProcessingError(RuntimeError):
    """Video decoding or extraction failed."""

class ApprovalError(ValueError):
    """Merchant approval cannot proceed with the selected references."""
