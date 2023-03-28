#!/usr/bin/env python


class InputError(Exception):
    """Raised when there is an error in the input."""

    pass


class UniqueError(Exception):
    """Raised when the dissimilarity matrix has only one unique entry."""

    pass
