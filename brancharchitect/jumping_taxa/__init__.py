__all__ = ['call_jumping_taxa']

# Use this pattern to avoid circular imports during static analysis
def __getattr__(name):
    if name == 'call_jumping_taxa':
        from brancharchitect.jumping_taxa.api import call_jumping_taxa as func
        return func
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")