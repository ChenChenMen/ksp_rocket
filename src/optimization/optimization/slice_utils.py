def resolve_slice(selection_index_collection: list[tuple[int, int] | int] = None) -> list[int] | None:
    """Resolve slice inputs into continuous slices."""
    # If no downselection is needed, return directly
    if selection_index_collection is None:
        return None

    # Resolve the indices from the collection
    selection_indices = []
    for element in selection_index_collection:
        if isinstance(element, int):
            selection_indices.append(element)
        elif isinstance(element, tuple):
            start_idx, end_idx = element
            selection_indices.extend(range(start_idx, end_idx))
    return selection_indices
