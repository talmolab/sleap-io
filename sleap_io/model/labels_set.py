"""Data model for collections of Labels objects."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, ItemsView, Iterator, KeysView, Union, ValuesView

import attrs

from sleap_io.model.labels import Labels


@attrs.define
class LabelsSet:
    """Container for multiple Labels objects with dictionary and tuple-like interface.

    This class provides a way to manage collections of Labels objects, such as
    train/val/test splits. It supports both dictionary-style access by name and
    tuple-style unpacking for backward compatibility.

    Attributes:
        labels: Dictionary mapping names to Labels objects.

    Examples:
        Create from existing Labels objects:
        >>> labels_set = LabelsSet({"train": train_labels, "val": val_labels})

        Access like a dictionary:
        >>> train = labels_set["train"]
        >>> for name, labels in labels_set.items():
        ...     print(f"{name}: {len(labels)} frames")

        Unpack like a tuple:
        >>> train, val = labels_set  # Order preserved from insertion

        Add new Labels:
        >>> labels_set["test"] = test_labels
    """

    labels: Dict[str, Labels] = attrs.field(factory=dict)

    def __getitem__(self, key: Union[str, int]) -> Labels:
        """Get Labels by name (string) or index (int) for tuple-like access.

        Args:
            key: Either a string name or integer index.

        Returns:
            The Labels object associated with the key.

        Raises:
            KeyError: If string key not found.
            IndexError: If integer index out of range.
        """
        if isinstance(key, int):
            try:
                return list(self.labels.values())[key]
            except IndexError:
                raise IndexError(
                    f"Index {key} out of range for LabelsSet with {len(self)} items"
                )
        return self.labels[key]

    def __setitem__(self, key: str, value: Labels) -> None:
        """Set a Labels object with a given name.

        Args:
            key: Name for the Labels object.
            value: Labels object to store.

        Raises:
            TypeError: If key is not a string or value is not a Labels object.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, not {type(key).__name__}")
        if not isinstance(value, Labels):
            raise TypeError(
                f"Value must be a Labels object, not {type(value).__name__}"
            )
        self.labels[key] = value

    def __delitem__(self, key: str) -> None:
        """Remove a Labels object by name.

        Args:
            key: Name of the Labels object to remove.

        Raises:
            KeyError: If key not found.
        """
        del self.labels[key]

    def __iter__(self) -> Iterator[Labels]:
        """Iterate over Labels objects (not keys) for tuple-like unpacking.

        This allows LabelsSet to be unpacked like a tuple:
        >>> train, val = labels_set

        Returns:
            Iterator over Labels objects in insertion order.
        """
        return iter(self.labels.values())

    def __len__(self) -> int:
        """Return the number of Labels objects."""
        return len(self.labels)

    def __contains__(self, key: str) -> bool:
        """Check if a named Labels object exists.

        Args:
            key: Name to check.

        Returns:
            True if the name exists in the set.
        """
        return key in self.labels

    def __repr__(self) -> str:
        """Return a string representation of the LabelsSet."""
        items = []
        for name, labels in self.labels.items():
            items.append(f"{name}: {len(labels)} labeled frames")
        items_str = ", ".join(items)
        return f"LabelsSet({items_str})"

    def keys(self) -> KeysView[str]:
        """Return a view of the Labels names."""
        return self.labels.keys()

    def values(self) -> ValuesView[Labels]:
        """Return a view of the Labels objects."""
        return self.labels.values()

    def items(self) -> ItemsView[str, Labels]:
        """Return a view of (name, Labels) pairs."""
        return self.labels.items()

    def get(self, key: str, default: Labels | None = None) -> Labels | None:
        """Get a Labels object by name with optional default.

        Args:
            key: Name of the Labels to retrieve.
            default: Default value if key not found.

        Returns:
            The Labels object or default if not found.
        """
        return self.labels.get(key, default)

    def save(
        self,
        save_dir: Union[str, Path],
        embed: Union[bool, str] = True,
        format: str = "slp",
        **kwargs,
    ) -> None:
        """Save all Labels objects to a directory.

        Args:
            save_dir: Directory to save the files to. Will be created if it
                doesn't exist.
            embed: For SLP format: Whether to embed images in the saved files.
                Can be True, False, "user", "predictions", or "all".
                See Labels.save() for details.
            format: Output format. Currently supports "slp" (default) and "ultralytics".
            **kwargs: Additional format-specific arguments. For ultralytics format,
                these might include skeleton, image_size, etc.

        Examples:
            Save as SLP files with embedded images:
            >>> labels_set.save("path/to/splits/", embed=True)

            Save as SLP files without embedding:
            >>> labels_set.save("path/to/splits/", embed=False)

            Save as Ultralytics dataset:
            >>> labels_set.save("path/to/dataset/", format="ultralytics")
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if format == "slp":
            for name, labels in self.items():
                if embed:
                    filename = f"{name}.pkg.slp"
                else:
                    filename = f"{name}.slp"
                labels.save(save_dir / filename, embed=embed)

        elif format == "ultralytics":
            # Import here to avoid circular imports
            from sleap_io.io import ultralytics

            # For ultralytics, we need to save each split in the proper structure
            for name, labels in self.items():
                # Map common split names
                split_name = name
                if name in ["training", "train"]:
                    split_name = "train"
                elif name in ["validation", "val", "valid"]:
                    split_name = "val"
                elif name in ["testing", "test"]:
                    split_name = "test"

                # Write this split
                ultralytics.write_labels(
                    labels, str(save_dir), split=split_name, **kwargs
                )

        else:
            raise ValueError(
                f"Unknown format: {format}. Supported formats: 'slp', 'ultralytics'"
            )

    @classmethod
    def from_labels_lists(
        cls, labels_list: list[Labels], names: list[str] | None = None
    ) -> LabelsSet:
        """Create a LabelsSet from a list of Labels objects.

        Args:
            labels_list: List of Labels objects.
            names: Optional list of names for the Labels. If not provided,
                will use generic names like "split1", "split2", etc.

        Returns:
            A new LabelsSet instance.

        Raises:
            ValueError: If names provided but length doesn't match labels_list.
        """
        if names is None:
            names = [f"split{i + 1}" for i in range(len(labels_list))]
        elif len(names) != len(labels_list):
            raise ValueError(
                f"Number of names ({len(names)}) must match number of Labels "
                f"({len(labels_list)})"
            )

        return cls(labels=dict(zip(names, labels_list)))
