#!/usr/bin/env python3
"""
Progress bar utilities for form-filler.

This module provides utility functions for creating and managing progress bars
during long-running operations in the form-filler tool.
"""

import time
from collections.abc import Callable
from typing import Any, TypeAlias, TypeVar, cast

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm
from tqdm.std import tqdm as tqdm_class

# Define a proper type for the tqdm class
ProgressType: TypeAlias = Progress | tqdm_class


def create_progress_bar(
    total: int,
    description: str = "Processing",
    use_rich: bool = True,
    transient: bool = True,
    leave: bool = False,
) -> ProgressType:
    """
    Create a progress bar for long-running operations.

    Args:
        total: Total number of steps or items to process
        description: Description to display before the progress bar
        use_rich: Whether to use rich progress bars (True) or tqdm (False)
        transient: Whether to remove the progress bar after completion (rich only)
        leave: Whether to leave the progress bar after completion (tqdm only)

    Returns:
        Progress bar object (either rich.Progress or tqdm)
    """
    if use_rich:
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=Console(),
            transient=transient,
        )
    else:
        return tqdm(
            total=total,
            desc=description,
            leave=leave,
            unit="step",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )


def create_indeterminate_spinner(description: str = "Processing") -> Progress:
    """
    Create an indeterminate spinner for operations with unknown duration.

    Args:
        description: Description to display before the spinner

    Returns:
        Rich Progress object with spinner
    """
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]{description}"),
        TimeElapsedColumn(),
        console=Console(),
        transient=True,
    )


T = TypeVar("T")


def with_progress_bar(
    iterable: list[T],
    description: str = "Processing",
    use_rich: bool = True,
    callback: Callable | None = None,
) -> list[T]:
    """
    Execute a function for each item in an iterable with a progress bar.

    Args:
        iterable: List of items to process
        description: Description to display before the progress bar
        use_rich: Whether to use rich progress bars (True) or tqdm (False)
        callback: Optional callback function to execute for each item

    Returns:
        List of results
    """
    results = []
    total = len(iterable)

    if use_rich:
        with create_progress_bar(total, description, use_rich=True) as progress:
            # Cast to Progress to make mypy happy
            rich_progress = cast("Progress", progress)
            task_id = rich_progress.add_task(description, total=total)
            for item in iterable:
                result = callback(item) if callback else item
                results.append(result)
                rich_progress.advance(task_id)
    else:
        with tqdm(
            total=total,
            desc=description,
            unit="item",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for item in iterable:
                result = callback(item) if callback else item
                results.append(result)
                pbar.update(1)

    return results


def wrap_process_with_progress(
    process_func: Callable,
    description: str = "Processing document",
    success_message: str = "✅ Document processed successfully!",
    error_message: str = "❌ Error processing document",
) -> Callable:
    """
    Wrap a processing function with a progress spinner.

    Args:
        process_func: Function to wrap
        description: Description to display during processing
        success_message: Message to display on success
        error_message: Message to display on error

    Returns:
        Wrapped function
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        console = Console()
        with create_indeterminate_spinner(description) as progress:
            # No need to track task_id since we're using progress.stop()
            progress.add_task(description, total=None)

            try:
                start_time = time.time()
                result = process_func(*args, **kwargs)
                elapsed = time.time() - start_time

                # Complete the progress
                progress.stop()

                if hasattr(result, "success") and result.success:
                    # Using the console defined above
                    console.print(f"{success_message} ({elapsed:.2f}s)")
                    return result
                else:
                    # Using the console defined above
                    console.print(f"{error_message}: {getattr(result, 'error', 'Unknown error')}")
                    return result
            except Exception as e:
                progress.stop()
                # Using the console defined above
                console.print(f"{error_message}: {e!s}")
                raise

    return wrapper
