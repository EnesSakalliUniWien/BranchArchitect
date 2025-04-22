# split.py
from typing import Set, Tuple, Optional, FrozenSet, Dict, TypeVar, Generic, Iterable, Self, cast, Iterator, List, Any, Union
from functools import total_ordering
from itertools import product
from pydantic_core.core_schema import set_schema
from collections.abc import MutableSet
from brancharchitect.partition import Partition


# Type variable for generic typing
T = TypeVar("T", bound="Partition")




@total_ordering
class FrozenPartitionSet(Generic[T]):
    """
    Immutable version of PartitionSet that can be used as dictionary keys.
    
    This class provides an immutable view of a set of Partitions with
    the same lookup. It supports set operations and comparisons.
    
    Attributes:
        _data: The underlying frozenset of Partitions
        _look_up: Mapping from string names to indices
        _reversed_lookup: Mapping from indices to string names
        _order: Optional ordering for the indices
        _name: Name of this partition set
    """

    def __init__(
        self,
        splits: Optional[Set[T]] = None,
        look_up: Optional[Dict[str, int]] = None,
        name: str = "FrozenPartitionSet",
        order: Optional[tuple] = None
    ) -> None:
        """
        Initialize a new FrozenPartitionSet.
        
        Args:
            splits: Set of Partitions to include
            look_up: Mapping from string names to indices
            name: Name of this partition set
            order: Optional ordering for the indices
        """
        self._data = frozenset(splits) if splits else frozenset()
        self._look_up = dict(look_up) if look_up else {}
        self._reversed_lookup : dict[int,str] = {v: k for k, v in self._look_up.items()}
        self._order = order if order is not None else (tuple(self._look_up.values()) if self._look_up else None)
        self._name : str = name

    def __contains__(self, x: object) -> bool:
        """Check if this partition set contains the given element."""
        return x in self._data

    def __iter__(self) -> Iterator[T]:
        """Iterate over the partitions in this set."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of partitions in this set."""
        return len(self._data)

    def __str__(self) -> str:
        """Return a string representation of this partition set."""
        splits_list = sorted(self._data, key=lambda s: s)
        return "\n".join(str(s) for s in splits_list)

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return str(sorted(self._data, key=lambda s: s))

    def __eq__(self, other: object) -> bool:
        """Check if this partition set equals another partition set."""
        if isinstance(other, (FrozenPartitionSet, PartitionSet)):
            return self._data == other._data
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        """Compare this partition set with another partition set."""
        if isinstance(other, (FrozenPartitionSet, PartitionSet)):
            return sorted(self._data) < sorted(other._data)
        return NotImplemented

    def __hash__(self) -> int:
        """Return a hash of this partition set."""
        return hash((self._data, self._order))

    def issubset(self, other: Iterable[T]) -> bool:
        """Return True if all elements in self are also in other."""
        if isinstance(other, (FrozenPartitionSet, PartitionSet)):
            return self._data.issubset(other._data)
        return self._data.issubset(other)

    def resolve_to_indices(self) -> List[Tuple[int, ...]]:
        """Return a list of tuples of indices for each partition."""
        return [s.indices for s in sorted(self._data, key=lambda p: sorted(p.indices))]


class PartitionSet(Generic[T], MutableSet[T]):
    """
    A set of partitions with common lookup information.
    
    This class represents a mutable set of Partition objects that share
    the same lookup information. It supports set operations, comparisons,
    and other standard set operations.
    
    Attributes:
        _data: The underlying set of Partitions
        look_up: Mapping from string names to indices
        reversed_lookup: Mapping from indices to string names
        order: Optional ordering for the indices
        name: Name of this partition set
    """
    
    def __init__(
        self,
        splits: Optional[Union[Set[T], Set[Partition]]] = None,
        look_up: Optional[Dict[str, int]] = None,
        name: str = "PartitionSet",
        order: Optional[tuple] = None
    ) -> None:
        """
        Initialize a new PartitionSet.
        
        Args:
            splits: Set of Partitions to include
            look_up: Mapping from string names to indices
            name: Name of this partition set
            order: Optional ordering for the indices
        """
        self._data: set[T] = set()
        
        self.look_up = look_up or {}
        self.reversed_lookup = {v: k for k, v in self.look_up.items()}
        self.order = order if order is not None else (tuple(self.look_up.values()) if self.look_up else None)
        self.name = name

        if splits:
            for split in splits:
                self.add(split)

        
        if not self.look_up:
            self.look_up = {str(i): i for i in range(len(self))}
            self.reversed_lookup = {v: k for k, v in self.look_up.items()}
            self.order = tuple(range(len(self)))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Generate a Pydantic schema for this class."""
        partition_schema = handler(Partition)
        return set_schema(partition_schema)
    
    def __contains__(self, x: object) -> bool:
        """Check if this partition set contains the given element."""
        return x in self._data

    def __iter__(self) -> Iterator[T]:
        """Iterate over the partitions in this set."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of partitions in this set."""
        return len(self._data)

    def add(self, element: Union[T, Tuple[int, ...], int, Partition]) -> None:
        """
        Add an element to this partition set.
        
        Args:
            element: The element to add (Partition, tuple, or int)
            
        Raises:
            TypeError: If the element is not a Partition, tuple, or int
        """
        if not isinstance(element, (Partition, tuple, int)):
            raise TypeError(f"Expected Partition/tuple/int but got {type(element)}")
        if isinstance(element, Partition):
            self._data.add(cast(T, element))  # Cast to T for type safety
        elif isinstance(element, tuple):
            self._data.add(cast(T, Partition(element, self.look_up or {})))  # Cast to T
        elif isinstance(element, int):
            self._data.add(cast(T, Partition((element,), self.look_up or {})))  # Cast to T
        else:
            raise TypeError(f"Can only add Partition objects or compatible types, got {type(element)}")

    def discard(self, element: T) -> None:
        """
        Remove an element from this partition set if it is present.
        
        Args:
            element: The element to remove
            
        Raises:
            ValueError: If the element is not in the set
        """
        if element not in self._data:
            raise ValueError(f"Element {element} not found in set")
        self._data.discard(element)

    def __hash__(self) -> int:
        """Return a hash of this partition set."""
        return hash((frozenset(self._data), self.order))

    def atom(self) -> Self:
        """Return a new PartitionSet containing only minimal elements."""
        try:
            minimal = {s for s in self if not any((other != s and other.taxa < s.taxa) for other in self)}
            return type(self)(minimal, look_up=self.look_up, name="atoms", order=self.order)
        except Exception as e:
            import sys
            print(f"Warning: Error in PartitionSet.atom: {e}", file=sys.stderr)
            return type(self)(set(), look_up=self.look_up, name="atoms_error", order=self.order)

    def cover(self) -> Self:
        """Return a new PartitionSet containing only maximal elements."""
        try:
            maximal = {s for s in self if not any((other != s and other.taxa > s.taxa) for other in self)}
            return type(self)(maximal, look_up=self.look_up, name="covering", order=self.order)
        except Exception as e:
            import sys
            print(f"Warning: Error in PartitionSet.cover: {e}", file=sys.stderr)
            return type(self)(set(), look_up=self.look_up, name="covering_error", order=self.order)

    def union(self, *others: Iterable[T]) -> Self:
        """
        Return a new PartitionSet with elements from the set and all others.
        
        Args:
            *others: Iterables to form union with
        
        Returns:
            A new PartitionSet containing the union
        """
        try:
            # First, collect all the sets into a list
            result = self._data.copy()  # Start with a copy of our data
            
            for other in others:
                if isinstance(other, PartitionSet):
                    result = result.union(other._data)
                else:
                    result = result.union(set(other))
            
            # Create a new PartitionSet with the result
            return type(self)(
                result, 
                look_up=self.look_up, 
                name=self.name + "_union", 
                order=self.order
            )
        except Exception as e:
            # Log the error but return a valid PartitionSet
            import sys
            print(f"Warning: Error in PartitionSet.union: {e}", file=sys.stderr)
            # Return a new PartitionSet with the current elements to avoid data loss
            return type(self)(self._data.copy(), look_up=self.look_up, name=self.name + "_union_error", order=self.order)

    def intersection(self, *others: Iterable[T]) -> Self:
        """
        Return a new PartitionSet with elements common to the set and all others.
        
        Args:
            *others: Iterables to intersect with
        
        Returns:
            A new PartitionSet containing the intersection
        """
        try:
            # First, collect all the sets into a list
            other_sets = []
            for other in others:
                if isinstance(other, PartitionSet):
                    other_sets.append(other._data)
                else:
                    other_sets.append(set(other))
            
            # Perform the intersection
            result = self._data.intersection(*other_sets)
            
            # Create a new PartitionSet with the result
            return type(self)(
                result, 
                look_up=self.look_up, 
                name=self.name + "_intersection", 
                order=self.order
            )
        except Exception as e:
            # Log the error but return a valid PartitionSet
            import sys
            print(f"Warning: Error in PartitionSet.intersection: {e}", file=sys.stderr)
            # Return a new empty PartitionSet with the same metadata
            return type(self)(set(), look_up=self.look_up, name=self.name + "_intersection_error", order=self.order)

    def difference(self, *others: Iterable[T]) -> Self:
        """
        Return a new PartitionSet with elements in this set but not in others.
        
        Args:
            *others: Iterables to exclude
        
        Returns:
            A new PartitionSet containing the difference
        """
        try:
            # First, collect all the sets into a list
            result = self._data.copy()  # Start with a copy of our data
            
            for other in others:
                if isinstance(other, PartitionSet):
                    result = result.difference(other._data)
                else:
                    result = result.difference(set(other))
            
            # Create a new PartitionSet with the result
            return type(self)(
                result, 
                look_up=self.look_up, 
                name=self.name + "_difference", 
                order=self.order
            )
        except Exception as e:
            # Log the error but return a valid PartitionSet
            import sys
            print(f"Warning: Error in PartitionSet.difference: {e}", file=sys.stderr)
            # Return a new PartitionSet with the current elements to avoid data loss
            return type(self)(self._data.copy(), look_up=self.look_up, name=self.name + "_difference_error", order=self.order)

    def symmetric_difference(self, other: Iterable[T]) -> Self:
        """
        Return a new PartitionSet with elements in either the set or other but not both.
        
        Args:
            other: Iterable to compare with
        
        Returns:
            A new PartitionSet containing the symmetric difference
        """
        try:
            # Perform the symmetric difference
            if isinstance(other, PartitionSet):
                result = self._data.symmetric_difference(other._data)
            else:
                result = self._data.symmetric_difference(set(other))
            
            # Create a new PartitionSet with the result
            return type(self)(
                result, 
                look_up=self.look_up, 
                name=self.name + "_symdiff", 
                order=self.order
            )
        except Exception as e:
            # Log the error but return a valid PartitionSet
            import sys
            print(f"Warning: Error in PartitionSet.symmetric_difference: {e}", file=sys.stderr)
            # Return a new PartitionSet with the current elements to avoid data loss
            return type(self)(self._data.copy(), look_up=self.look_up, name=self.name + "_symdiff_error", order=self.order)

    def __or__(self, other: object) -> Self:
        """Implement the | operator (union)."""
        if isinstance(other, Iterable):
            return self.union(other)
        return NotImplemented

    def __and__(self, other: object) -> Self:
        """Implement the & operator (intersection)."""
        if isinstance(other, Iterable):
            return self.intersection(other)
        return NotImplemented

    def __sub__(self, other: object) -> Self:
        """Implement the - operator (difference)."""
        if isinstance(other, Iterable):
            return self.difference(other)
        return NotImplemented

    def __xor__(self, other: object) -> Self:
        """Implement the ^ operator (symmetric_difference)."""
        if isinstance(other, Iterable):
            return self.symmetric_difference(other)
        return NotImplemented

    def __str__(self) -> str:
        """Return a string representation of this partition set."""
        splits_list = sorted(self, key=lambda s: s)
        return "\n".join(str(s) for s in splits_list)

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return str(sorted(self, key=lambda s: s))

    def __getitem__(self, index: int) -> T:
        """Get the partition at the specified index."""
        return sorted(self, key=lambda s: s)[index]

    def cartesian(self, other: "PartitionSet[T]") -> Self:
        """
        Return a new PartitionSet with the Cartesian product of this set and another.
        
        Args:
            other: Another PartitionSet to form the product with
            
        Returns:
            A new PartitionSet containing the Cartesian product
            
        Raises:
            TypeError: If other is not a PartitionSet
            ValueError: If the sets don't have compatible orders
        """
        try:
            if not isinstance(other, PartitionSet):
                raise TypeError("Cartesian product only defined for PartitionSet")
            if self.order is None or other.order is None:
                raise ValueError("Both PartitionSet objects must have an order to compute Cartesian product")
            if self.order != other.order:
                raise ValueError("PartitionSet objects must have the same order to compute Cartesian product")
                
            # Compute product of self and other, then merge each pair into a new Partition.
            new_partitions = {
                Partition(tuple(sorted(set(p1.indices) | set(p2.indices))), self.look_up or {})
                for p1, p2 in product(self, other)
            }
            
            return type(self)(
                new_partitions, 
                look_up=self.look_up, 
                order=self.order, 
                name=f"{self.name}_cartesian_{other.name}"
            )
        except Exception as e:
            # Log the error but return a valid PartitionSet
            import sys
            print(f"Warning: Error in PartitionSet.cartesian: {e}", file=sys.stderr)
            # Return a new empty PartitionSet with the same metadata
            return type(self)(set(), look_up=self.look_up, name=self.name + "_cartesian_error", order=self.order)

    def __mul__(self, other: "PartitionSet[T]") -> Self:
        """Implement the * operator (Cartesian product)."""
        return self.cartesian(other)
        
    def resolve_to_indices(self) -> List[Tuple[int, ...]]:
        """
        Convert the PartitionSet to a list of tuples of indices.
        
        Returns:
            A list of tuples where each tuple contains the indices from a Partition in this set.
        """
        try:
            result = []
            for s in sorted(self, key=lambda p: p.indices):
                result.append(s.resolve_to_indices())
            return result
        except Exception as e:
            # Log the error but return a valid result
            import sys
            print(f"Warning: Error in PartitionSet.resolve_to_indices: {e}", file=sys.stderr)
            return []
    
    @property
    def list_taxa_name(self) -> List[Tuple[str, ...]]:
        """Return a list of tuples of taxa names for each partition."""
        try:
            return [tuple(sorted(p.taxa)) for p in sorted(self, key=lambda p: sorted(p.taxa))]
        except Exception as e:
            # Log the error but return a valid result
            import sys
            print(f"Warning: Error in PartitionSet.list_taxa_name: {e}", file=sys.stderr)
            return []

    def issubset(self, other: Iterable[T]) -> bool:
        """
        Return True if all elements in self are also in other.
        
        Args:
            other: Iterable to check against
            
        Returns:
            True if self is a subset of other, False otherwise
        """
        if isinstance(other, PartitionSet):
            return self._data.issubset(other._data)
        elif isinstance(other, (set, frozenset)):
            return self._data.issubset(other)
        return all(elem in other for elem in self)
    
    def __le__(self, other: Iterable[T]) -> bool:
        """Return True if self <= other (issubset)."""
        return self.issubset(other)
    
    def __lt__(self, other: Iterable[T]) -> bool:
        """Return True if self < other (strict subset)."""
        return self.issubset(other) and self != other
    
    def issuperset(self, other: Iterable[T]) -> bool:
        """
        Return True if all elements in other are also in self.
        
        Args:
            other: Iterable to check against
            
        Returns:
            True if self is a superset of other, False otherwise
        """
        if isinstance(other, PartitionSet):
            return self._data.issuperset(other._data)
        elif isinstance(other, (set, frozenset)):
            return self._data.issuperset(other)
        return all(elem in self for elem in other)
    
    def __ge__(self, other: Iterable[T]) -> bool:
        """Return True if self >= other (issuperset)."""
        return self.issuperset(other)
    
    def __gt__(self, other: Iterable[T]) -> bool:
        """Return True if self > other (strict superset)."""
        return self.issuperset(other) and self != other
    
    @classmethod
    def from_existing(cls, source: Self, elements: Optional[Set[Any]] = None, name: Optional[str] = None) -> Self:
        """
        Create a new PartitionSet from an existing one, optionally with different elements.
        
        This method preserves important attributes like look_up and order from the source PartitionSet.
        """
        try:
            # Use a more explicit approach to convert the type
            if elements is None:
                # Create a new set of the correct type by iterating
                elements_to_use: Set[T] = set()
                for item in source._data:
                    elements_to_use.add(cast(T, item))
            else:
                elements_to_use = cast(Set[T], elements)
                
            name_to_use = source.name if name is None else name
            
            result = cls(elements_to_use, look_up=source.look_up, name=name_to_use, order=source.order)
            return result
        except Exception as e:
            # Log the error but return a valid PartitionSet
            import sys
            print(f"Warning: Error in PartitionSet.from_existing: {e}", file=sys.stderr)
            # Return a new empty PartitionSet with the source metadata
            return cls(set(), look_up=source.look_up, name=name_to_use or "from_existing_error", order=source.order)
    
    def copy(self, name: Optional[str] = None) -> Self:
        """
        Return a shallow copy of this PartitionSet, preserving its 'look_up',
        'order', and elements. A custom 'name' can be provided if desired.
        
        Args:
            name: Optional new name for the copy
            
        Returns:
            A new PartitionSet with the same elements and metadata
        """
        try:
            return type(self).from_existing(self, name=name)
        except Exception as e:
            # Log the error but return a valid PartitionSet
            import sys
            print(f"Warning: Error in PartitionSet.copy: {e}", file=sys.stderr)
            # Return a new PartitionSet with the current elements
            return type(self)(self._data.copy(), look_up=self.look_up, name=name or self.name + "_copy_error", order=self.order)
    
    def freeze(self) -> FrozenPartitionSet[T]:
        """
        Convert to an immutable FrozenPartitionSet.
        
        Returns:
            A new FrozenPartitionSet with the same elements and metadata
        """
        try:
            return FrozenPartitionSet(
                splits=self._data,
                look_up=self.look_up,
                name=self.name + "_frozen",
                order=self.order
            )
        except Exception as e:
            # Log the error but return a valid FrozenPartitionSet
            import sys
            print(f"Warning: Error in PartitionSet.freeze: {e}", file=sys.stderr)
            # Return an empty FrozenPartitionSet with the same metadata
            return FrozenPartitionSet(
                splits=set(),
                look_up=self.look_up,
                name=self.name + "_frozen_error",
                order=self.order
            )