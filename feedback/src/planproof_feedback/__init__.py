"""Optional network feedback. Importing this package never calls a provider."""

from .service import FeedbackService, create_service

__all__ = ["FeedbackService", "create_service"]
