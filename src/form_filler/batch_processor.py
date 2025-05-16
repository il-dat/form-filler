#!/usr/bin/env python3
"""
CrewAI Batch processor for multiple Vietnamese documents.

Processes all documents in a directory and fills forms using CrewAI agents.
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Import the CrewAI document processor
from form_filler.crew import DocumentProcessingCrew


@dataclass
class BatchJob:
    """Represents a single batch processing job."""

    source_path: Path
    form_path: Path
    output_path: Path
    status: str = "pending"  # pending, processing, completed, failed
    start_time: float = 0
    end_time: float = 0
    error: str = ""
    crew_id: str = ""


class CrewAIBatchProcessor:
    """Handles batch processing of multiple documents using CrewAI."""

    def __init__(  # noqa: D107
        self,
        text_model: str = "llama3.2:3b",
        extraction_method: str = "traditional",
        vision_model: str = "llava:7b",
        max_concurrent: int = 3,
        timeout: int = 300,
    ):
        self.text_model = text_model
        self.extraction_method = extraction_method
        self.vision_model = vision_model
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.jobs: list[BatchJob] = []

        # Setup logging and console
        self.logger = logging.getLogger("CrewAIBatchProcessor")
        self.console = Console()

    def add_job(self, source_path: str, form_path: str, output_path: str) -> BatchJob:
        """Add a job to the batch."""
        job = BatchJob(
            source_path=Path(source_path),
            form_path=Path(form_path),
            output_path=Path(output_path),
            crew_id=f"crew_{len(self.jobs)}",
        )
        self.jobs.append(job)
        return job

    def discover_jobs(
        self,
        source_dir: str,
        form_template: str,
        output_dir: str,
        pattern: str = "**/*.pdf",
    ) -> int:
        """Automatically discover documents to process."""
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        form_path = Path(form_template)

        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all matching files
        files = list(source_path.glob(pattern))

        for file_path in files:
            # Generate output filename
            stem = file_path.stem
            output_file = output_path / f"{stem}_filled.docx"

            self.add_job(str(file_path), str(form_path), str(output_file))

        return len(files)

    def process_single_job(self, job: BatchJob) -> BatchJob:
        """Process a single job using CrewAI."""
        job.status = "processing"
        job.start_time = time.time()

        try:
            self.logger.info(f"Processing: {job.source_path.name} (Crew: {job.crew_id})")

            # Create CrewAI processor for this job
            crew_processor = DocumentProcessingCrew(
                text_model=self.text_model,
                extraction_method=self.extraction_method,
                vision_model=self.vision_model,
            )

            # Process the document
            result = crew_processor.process_document(
                str(job.source_path),
                str(job.form_path),
                str(job.output_path),
            )

            if result.success:
                job.status = "completed"
                self.logger.info(f"✅ Completed: {job.source_path.name} (Crew: {job.crew_id})")
            else:
                job.status = "failed"
                job.error = str(result.error) if result.error is not None else ""
                self.logger.error(
                    f"❌ Failed: {job.source_path.name} (Crew: {job.crew_id}) - {result.error}",
                )

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self.logger.error(f"💥 Error: {job.source_path.name} (Crew: {job.crew_id}) - {e}")

        finally:
            job.end_time = time.time()

        return job

    def process_all(
        self, progress_callback: Callable[[int, int, BatchJob], None] | None = None
    ) -> dict[str, Any]:
        """Process all jobs in the batch using ThreadPoolExecutor."""
        if not self.jobs:
            return {"total": 0, "completed": 0, "failed": 0}

        start_time = time.time()
        self.logger.info(f"Starting CrewAI batch processing of {len(self.jobs)} jobs")
        self.logger.info(
            f"Configuration: {self.text_model} | {self.extraction_method} | {self.max_concurrent} workers",
        )

        # Process jobs with ThreadPoolExecutor for true parallelism
        completed_jobs = []

        # Create rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[bold]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            overall_task = progress.add_task(
                f"[bold]Processing {len(self.jobs)} documents...",
                total=len(self.jobs),
            )

            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                # Submit all jobs
                future_to_job = {
                    executor.submit(self.process_single_job, job): job for job in self.jobs
                }

                # Process completed jobs
                for future in as_completed(future_to_job, timeout=self.timeout * len(self.jobs)):
                    try:
                        job = future.result()
                        completed_jobs.append(job)

                        # Update the progress bar
                        progress.update(
                            overall_task,
                            advance=1,
                            description=f"[bold]Processing {len(completed_jobs)}/{len(self.jobs)} documents...",
                        )

                        # Status updated via progress bar - no need for additional variables here

                        # Call the original callback if provided
                        if progress_callback:
                            progress_callback(len(completed_jobs), len(self.jobs), job)

                    except Exception as e:
                        job = future_to_job[future]
                        job.status = "failed"
                        job.error = f"Execution error: {e}"
                        job.end_time = time.time()
                        completed_jobs.append(job)
                        self.logger.error(f"Job execution failed: {job.source_path.name} - {e}")

                        # Update the progress bar for failures too
                        progress.update(
                            overall_task,
                            advance=1,
                            description=f"[bold]Processing {len(completed_jobs)}/{len(self.jobs)} documents...",
                        )

        end_time = time.time()

        # Generate statistics
        stats = self.generate_statistics(end_time - start_time)
        self.logger.info(f"CrewAI batch processing completed in {stats['total_time']:.2f}s")

        return stats

    def generate_statistics(self, total_time: float) -> dict:
        """Generate processing statistics."""
        total = len(self.jobs)
        completed = sum(1 for job in self.jobs if job.status == "completed")
        failed = sum(1 for job in self.jobs if job.status == "failed")

        # Calculate average processing time for completed jobs
        completed_jobs = [job for job in self.jobs if job.status == "completed"]
        avg_time = 0.0
        if completed_jobs:
            total_job_time = sum(job.end_time - job.start_time for job in completed_jobs)
            avg_time = total_job_time / len(completed_jobs) if completed_jobs else 0.0

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "success_rate": (completed / total * 100) if total > 0 else 0,
            "total_time": total_time,
            "average_job_time": avg_time,
            "jobs_per_second": float(completed) / total_time if total_time > 0 else 0.0,
            "extraction_method": self.extraction_method,
            "text_model": self.text_model,
            "vision_model": self.vision_model if self.extraction_method == "ai" else None,
        }

    def save_report(self, output_path: str, stats: dict[str, Any]) -> None:
        """Save detailed processing report."""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {
                "text_model": self.text_model,
                "extraction_method": self.extraction_method,
                "vision_model": self.vision_model,
                "max_concurrent": self.max_concurrent,
                "timeout": self.timeout,
            },
            "statistics": stats,
            "jobs": [
                {
                    "crew_id": job.crew_id,
                    "source": str(job.source_path),
                    "output": str(job.output_path),
                    "status": job.status,
                    "processing_time": job.end_time - job.start_time if job.end_time > 0 else 0,
                    "error": job.error,
                }
                for job in self.jobs
            ],
        }

        with Path(output_path).open("w") as f:
            json.dump(report, f, indent=2)


# CLI Commands for CrewAI batch processing
@click.group()
@click.option(
    "--text-model",
    "-tm",
    default="llama3.2:3b",
    help="Ollama text model for translation and form filling",
)
@click.option(
    "--extraction-method",
    "-e",
    type=click.Choice(["traditional", "ai"]),
    default="traditional",
    help="Text extraction method",
)
@click.option("--vision-model", "-vm", default="llava:7b", help="Vision model for AI extraction")
@click.option("--max-concurrent", "-c", default=3, help="Maximum concurrent CrewAI processes")
@click.option("--timeout", "-t", default=300, help="Timeout per job in seconds")
@click.pass_context
def batch_cli(
    ctx: click.Context,
    text_model: str,
    extraction_method: str,
    vision_model: str,
    max_concurrent: int,
    timeout: int,
) -> None:
    """CrewAI batch processing commands for Vietnamese document form filling.

    All batch commands display enhanced progress bars with real-time updates on the status
    of each job in the batch. You can see the overall progress, estimated time remaining,
    and per-job status with completion indicators.
    """
    ctx.ensure_object(dict)
    ctx.obj["text_model"] = text_model
    ctx.obj["extraction_method"] = extraction_method
    ctx.obj["vision_model"] = vision_model
    ctx.obj["max_concurrent"] = max_concurrent
    ctx.obj["timeout"] = timeout


@batch_cli.command()
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("form_template", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option("--pattern", "-p", default="**/*.pdf", help="File pattern to match")
@click.option("--report", "-r", type=click.Path(), help="Save processing report")
@click.pass_context
def process_directory(
    ctx: click.Context,
    source_dir: str,
    form_template: str,
    output_dir: str,
    pattern: str,
    report: str | None,
) -> None:
    """Process all documents in a directory using CrewAI.

    SOURCE_DIR: Directory containing Vietnamese documents
    FORM_TEMPLATE: DOCX form template to fill
    OUTPUT_DIR: Directory to save filled forms

    Example with AI extraction and multiple crews:
    python crew_batch_processor.py -e ai -vm llava:7b -c 5 process-directory input/ form.docx output/
    """
    processor = CrewAIBatchProcessor(
        text_model=ctx.obj["text_model"],
        extraction_method=ctx.obj["extraction_method"],
        vision_model=ctx.obj["vision_model"],
        max_concurrent=ctx.obj["max_concurrent"],
        timeout=ctx.obj["timeout"],
    )

    # Discover jobs
    count = processor.discover_jobs(source_dir, form_template, output_dir, pattern)
    click.echo(f"Found {count} documents to process")
    click.echo("CrewAI Configuration:")
    click.echo(f"  - Text model: {ctx.obj['text_model']}")
    click.echo(f"  - Extraction method: {ctx.obj['extraction_method']}")
    if ctx.obj["extraction_method"] == "ai":
        click.echo(f"  - Vision model: {ctx.obj['vision_model']}")
    click.echo(f"  - Max concurrent crews: {ctx.obj['max_concurrent']}")

    if count == 0:
        click.echo("No documents found matching the pattern")
        return

    # Progress tracking with rich display
    def progress_callback(completed: int, total: int, job: BatchJob) -> None:
        # This function is replaced by the rich progress bar, but we keep it
        # for backward compatibility with existing code
        pass

    # Process all jobs
    click.echo("\nStarting CrewAI batch processing...")
    stats = processor.process_all(progress_callback)

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("CREWAI BATCH PROCESSING RESULTS")
    click.echo("=" * 60)
    click.echo(f"Total jobs: {stats['total']}")
    click.echo(f"Completed: {stats['completed']} ({stats['success_rate']:.1f}%)")
    click.echo(f"Failed: {stats['failed']}")
    click.echo(f"Total time: {stats['total_time']:.2f} seconds")
    click.echo(f"Average per job: {stats['average_job_time']:.2f} seconds")
    click.echo(f"Throughput: {stats['jobs_per_second']:.2f} jobs/second")
    click.echo(f"Extraction method: {stats['extraction_method']}")
    click.echo(f"Text model: {stats['text_model']}")
    if stats.get("vision_model"):
        click.echo(f"Vision model: {stats['vision_model']}")

    # Save report if requested
    if report:
        processor.save_report(report, stats)
        click.echo(f"\nDetailed report saved to: {report}")

    # Show failed jobs
    failed_jobs = [job for job in processor.jobs if job.status == "failed"]
    if failed_jobs:
        click.echo(f"\nFailed jobs ({len(failed_jobs)}):")
        for job in failed_jobs:
            click.echo(f"- {job.source_path.name} (Crew: {job.crew_id}): {job.error}")


@batch_cli.command()
@click.argument("jobs_file", type=click.Path(exists=True))
@click.option("--report", "-r", type=click.Path(), help="Save processing report")
@click.pass_context
def process_from_file(ctx: click.Context, jobs_file: str, report: str | None) -> None:
    """Process jobs defined in a JSON file using CrewAI.

    JOBS_FILE: JSON file containing job definitions
    """
    processor = CrewAIBatchProcessor(
        text_model=ctx.obj["text_model"],
        extraction_method=ctx.obj["extraction_method"],
        vision_model=ctx.obj["vision_model"],
        max_concurrent=ctx.obj["max_concurrent"],
        timeout=ctx.obj["timeout"],
    )

    # Load jobs from file
    with Path(jobs_file).open() as f:
        jobs_data: dict[str, Any] = json.load(f)

    for job_data in jobs_data.get("jobs", []):
        processor.add_job(job_data["source"], job_data["form"], job_data["output"])

    click.echo(f"Loaded {len(processor.jobs)} jobs from {jobs_file}")

    # Progress tracking with rich display
    def progress_callback(completed: int, total: int, job: BatchJob) -> None:
        # This function is replaced by the rich progress bar, but we keep it
        # for backward compatibility with existing code
        pass

    # Process all jobs
    stats = processor.process_all(progress_callback)

    # Display results
    click.echo(f"\nCompleted: {stats['completed']}/{stats['total']} jobs")
    click.echo(f"Success rate: {stats['success_rate']:.1f}%")
    click.echo(f"Using {stats['extraction_method']} extraction with {stats['text_model']}")

    # Save report if requested
    if report:
        processor.save_report(report, stats)
        click.echo(f"Report saved to: {report}")


@batch_cli.command()
@click.argument("output_file", type=click.Path())
@click.option("--source-dir", required=True, help="Source directory")
@click.option("--form", required=True, help="Form template")
@click.option("--output-dir", required=True, help="Output directory")
@click.option("--pattern", default="**/*.pdf", help="File pattern")
def generate_jobs_file(
    output_file: str, source_dir: str, form: str, output_dir: str, pattern: str
) -> None:
    """Generate a jobs file for CrewAI batch processing.

    OUTPUT_FILE: Path to save the jobs JSON file
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Find all matching files
    files = list(source_path.glob(pattern))

    jobs = []
    for file_path in files:
        stem = file_path.stem
        output_file_path = output_path / f"{stem}_filled.docx"

        jobs.append({"source": str(file_path), "form": form, "output": str(output_file_path)})

    # Create jobs file
    jobs_data = {
        "description": f"CrewAI batch processing jobs for {len(jobs)} documents",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_jobs": len(jobs),
        "configuration": {
            "recommended_concurrent": min(len(jobs), 5),
            "estimated_time_per_job": "60-120 seconds",
            "extraction_methods": ["traditional", "ai"],
        },
        "jobs": jobs,
    }

    with Path(output_file).open("w") as f:
        json.dump(jobs_data, f, indent=2)

    click.echo(f"Generated CrewAI jobs file with {len(jobs)} jobs: {output_file}")
    # Check if the configuration contains the recommended_concurrent field
    if (
        isinstance(jobs_data, dict)
        and "configuration" in jobs_data
        and isinstance(jobs_data["configuration"], dict)
    ):
        recommended = jobs_data["configuration"].get("recommended_concurrent", 1)
        click.echo(f"Recommended concurrent crews: {recommended}")
    else:
        click.echo("Recommended concurrent crews: 1")


@batch_cli.command()
def crew_status() -> None:
    """Check CrewAI and Ollama status for batch processing."""
    click.echo("🔍 Checking CrewAI Batch Processing Status...")

    # Check CrewAI installation
    try:
        import crewai

        click.echo(f"✅ CrewAI installed: v{crewai.__version__}")
    except ImportError:
        click.echo("❌ CrewAI not installed. Run: pip install crewai crewai-tools")
        return

    # Check Ollama connection
    import aiohttp

    async def check_ollama() -> bool:
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    "http://localhost:11434/api/tags",
                ) as response,
            ):
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])

                    text_models = [
                        m
                        for m in models
                        if not any(v in m["name"].lower() for v in ["llava", "vision", "bakllava"])
                    ]
                    vision_models = [
                        m
                        for m in models
                        if any(v in m["name"].lower() for v in ["llava", "vision", "bakllava"])
                    ]

                    click.echo(f"✅ Ollama running with {len(models)} models")
                    click.echo(
                        f"  - Text models: {len(text_models)} (for translation/form filling)",
                    )
                    click.echo(f"  - Vision models: {len(vision_models)} (for AI extraction)")

                    if len(text_models) == 0:
                        click.echo(
                            "⚠️  No text models found. Install with: ollama pull llama3.2:3b",
                        )
                    if len(vision_models) == 0:
                        click.echo(
                            "⚠️  No vision models found. Install with: ollama pull llava:7b",
                        )

                    return True
                else:
                    click.echo(f"❌ Ollama returned status: {response.status}")
                    return False
        except Exception as e:
            click.echo(f"❌ Cannot connect to Ollama: {e}")
            return False

    # Run async check

    ollama_ok = asyncio.run(check_ollama())

    if ollama_ok:
        click.echo("\n🚀 CrewAI Batch Processing is ready!")
        click.echo("💡 Tips for optimal performance:")
        click.echo("  - Use 3-5 concurrent crews for balanced performance")
        click.echo("  - AI extraction is slower but more accurate for complex documents")
        click.echo("  - Traditional extraction is faster for clean, text-based documents")
    else:
        click.echo("\n❌ CrewAI Batch Processing not ready. Please fix Ollama connection.")


if __name__ == "__main__":
    batch_cli()
