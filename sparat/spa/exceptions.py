from nengo.exceptions import NengoException


class SpaException(NengoException):
    """A exception within the SPA subsystem."""


class SpaModuleError(SpaException, ValueError):
    """An error in how SPA keeps track of modules."""


class SpaParseError(SpaException, ValueError):
    """An error encountered while parsing a SPA expression."""


class SpaTypeError(SpaException, ValueError):
    """The evaluation of types in an SPA expression was invalid."""
