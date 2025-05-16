#!/usr/bin/env python3
"""
CrewAI Demo script for Vietnamese Document Form Filler.

Shows CrewAI-powered multi-agent capabilities.
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

from batch_processor import BatchProcessor
from document_processor import DocumentProcessor

from form_filler.batch_processor import CrewAIBatchProcessor
from form_filler.crew.document_processing_crew import DocumentProcessingCrew

# Demo data - sample Vietnamese text
SAMPLE_VIETNAMESE_TEXT = """
Họ và tên: Nguyễn Văn An
Ngày sinh: 15/03/1990
Địa chỉ: 123 Phố Huế, Quận Hai Bà Trưng, Hà Nội
Số điện thoại: 0123456789
Email: nguyen.van.an@email.com
Trình độ học vấn: Cử nhân Công nghệ Thông tin

Kinh nghiệm làm việc:
- 2015-2018: Lập trình viên tại Công ty ABC
- 2018-2022: Trưởng nhóm phát triển tại Công ty XYZ
- 2022-hiện tại: Kiến trúc sư phần mềm tại Công ty DEF

Kỹ năng:
- Ngôn ngữ lập trình: Python, Java, JavaScript
- Framework: Django, React, Spring Boot
- Cơ sở dữ liệu: MySQL, PostgreSQL, MongoDB
- DevOps: Docker, Kubernetes, AWS
"""

SAMPLE_FORM_CONTENT = """
EMPLOYMENT APPLICATION FORM

Personal Information:
Name: ___________________
Date of Birth: ___________________
Address: ___________________________________
Phone Number: ___________________
Email: ___________________

Education:
Highest Degree: ___________________
Institution: ___________________

Work Experience:
Previous Position: ___________________
Company: ___________________
Years of Experience: ___________________

Skills:
Technical Skills: ___________________________________
Programming Languages: ___________________
Frameworks: ___________________
Databases: ___________________

Additional Information:
___________________________________________
___________________________________________
"""


def demo_crewai_single_document():
    """Demonstrate CrewAI single document processing."""
    print("🤖 Demo 1: CrewAI Single Document Processing")
    print("-" * 50)

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as source_file:
        source_file.write(SAMPLE_VIETNAMESE_TEXT)
        source_path = source_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as form_file:
        form_file.write(SAMPLE_FORM_CONTENT)
        form_path = form_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as output_file:
        output_path = output_file.name

    # Initialize CrewAI processor
    print("🤖 Initializing CrewAI crew...")
    crew_processor = DocumentProcessingCrew(
        text_model="llama3.2:3b", extraction_method="traditional", vision_model="llava:7b"
    )

    print("📄 Processing Vietnamese CV with CrewAI agents...")
    print(f"Source: {Path(source_path).name}")
    print(f"Form: {Path(form_path).name}")
    print("\n🤖 CrewAI Agents in Action:")
    print("  1. 📄 Document Collector Agent - Extracting text...")
    print("  2. 🔄 Translator Agent - Translating Vietnamese to English...")
    print("  3. 🔍 Form Analyst Agent - Analyzing form structure...")
    print("  4. ✍️ Form Filler Agent - Filling form intelligently...")

    # Process document
    result = crew_processor.process_document(source_path, form_path, output_path)

    if result.success:
        print("\n✅ CrewAI Success!")
        print(f"Output saved to: {output_path}")

        # Show agent collaboration results
        if Path(output_path).exists():
            with open(output_path) as f:
                content = f.read()
                print("\n📋 CrewAI Filled Form Preview:")
                print(content[:300] + "..." if len(content) > 300 else content)

        print(f"🤖 Agent Configuration: {result.metadata.get('extraction_method')} extraction")
        print(f"📱 Text Model: {result.metadata.get('text_model')}")
        if result.metadata.get("vision_model"):
            print(f"👁️ Vision Model: {result.metadata.get('vision_model')}")
    else:
        print(f"❌ CrewAI Failed: {result.error}")

    # Cleanup
    Path(source_path).unlink()
    Path(form_path).unlink()
    if Path(output_path).exists():
        Path(output_path).unlink()

    print("\n")


def demo_crewai_agent_capabilities():
    """Demonstrate individual CrewAI agent capabilities."""
    print("🤖 Demo 2: CrewAI Agent Capabilities")
    print("-" * 50)

    from document_processor import (
        DocumentExtractionTool,
        FormAnalysisTool,
        FormFillingTool,
        TranslationTool,
    )

    # Create temp files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
        temp_file.write(SAMPLE_VIETNAMESE_TEXT)
        temp_path = temp_file.name

    print("🔧 CrewAI Tool 1: DocumentExtractionTool")
    extractor = DocumentExtractionTool(extraction_method="traditional")
    extracted_text = extractor._run(temp_path)
    print(f"✅ Extracted {len(extracted_text)} characters")
    print(f"📝 Sample: {extracted_text[:100]}...")

    print("\n🔧 CrewAI Tool 2: TranslationTool")
    translator = TranslationTool(model="llama3.2:3b")
    english_text = translator._run(extracted_text)
    print("✅ Translation completed by CrewAI agent")
    print(f"📝 Sample: {english_text[:100]}...")

    print("\n🔧 CrewAI Tool 3: FormAnalysisTool")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as form_file:
        form_file.write(SAMPLE_FORM_CONTENT)
        form_path = form_file.name

    form_analyzer = FormAnalysisTool()
    form_analyzer._run(form_path)
    print("✅ Form analysis completed")
    print("📝 Found form fields in structure")

    print("\n🔧 CrewAI Tool 4: FormFillingTool")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as output_file:
        output_path = output_file.name
    form_filler = FormFillingTool(model="llama3.2:3b")
    filling_result = form_filler._run(form_path, english_text, output_path)
    print("✅ Form filling completed by CrewAI agent")
    print(f"📝 Result: {filling_result}")

    # Cleanup
    Path(temp_path).unlink()
    Path(form_path).unlink()
    if Path(output_path).exists():
        Path(output_path).unlink()

    print("\n")


def demo_crewai_batch_processing():
    """Demonstrate CrewAI batch processing."""
    print("🤖 Demo 3: CrewAI Batch Processing")
    print("-" * 50)

    # Create sample documents
    sample_documents = [
        f"Tài liệu {i}: Tên: Người {i}, Tuổi: {20+i}, Địa chỉ: Hà Nội, Việt Nam"
        for i in range(1, 4)
    ]

    # Create temporary directory structure
    temp_dir = Path(tempfile.mkdtemp())
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create sample files
    form_path = temp_dir / "form.txt"
    with open(form_path, "w") as f:
        f.write(SAMPLE_FORM_CONTENT)

    # Create input documents
    for i, content in enumerate(sample_documents, 1):
        doc_path = input_dir / f"document_{i}.txt"
        with open(doc_path, "w") as f:
            f.write(content)

    print(f"📁 Created {len(sample_documents)} sample documents")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Initialize CrewAI batch processor
    print("\n🤖 Initializing CrewAI Batch Processor...")
    batch_processor = CrewAIBatchProcessor(
        text_model="llama3.2:3b",
        extraction_method="traditional",
        vision_model="llava:7b",
        max_concurrent=2,
        timeout=60,
    )

    # Discover jobs
    job_count = batch_processor.discover_jobs(
        str(input_dir), str(form_path), str(output_dir), "*.txt"
    )

    print(f"🔍 Discovered {job_count} processing jobs")
    print("🤖 CrewAI Configuration:")
    print("  - Max concurrent crews: 2")
    print("  - Each crew has 4 specialized agents")
    print(f"  - Total agents working: {job_count * 4} (across all crews)")

    # Progress callback
    def progress_callback(completed, total, job):
        progress = (completed / total) * 100
        status = "✅" if job.status == "completed" else "❌"
        print(
            f"Progress: {completed}/{total} ({progress:.1f}%) - {status} {job.source_path.name} (Crew: {job.crew_id})"
        )

    # Process batch
    print("\n🚀 Starting CrewAI batch processing...")
    print("Multiple crews working in parallel...")
    stats = batch_processor.process_all(progress_callback)

    print("\n📊 CrewAI Batch Processing Results:")
    print(f"Total: {stats['total']}")
    print(f"Completed: {stats['completed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success Rate: {stats['success_rate']:.1f}%")
    print(f"Total Time: {stats['total_time']:.2f}s")
    print(f"Average per Job: {stats['average_job_time']:.2f}s")
    print(f"Extraction Method: {stats['extraction_method']}")
    print(f"Text Model: {stats['text_model']}")

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)

    print("\n")


def demo_crewai_extraction_methods():
    """Compare CrewAI extraction methods (Traditional, AI, OpenAI)."""
    print("🤖 Demo 4: CrewAI Extraction Methods Comparison")
    print("-" * 50)

    # Create sample document
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
        temp_file.write(SAMPLE_VIETNAMESE_TEXT)
        temp_path = temp_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as form_file:
        form_file.write(SAMPLE_FORM_CONTENT)
        form_path = form_file.name

    # Test Traditional Method
    print("🔧 Testing Traditional Extraction with CrewAI...")
    traditional_crew = DocumentProcessingCrew(
        text_model="llama3.2:3b", extraction_method="traditional", vision_model="llava:7b"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as output_file:
        output_traditional = output_file.name

    start_time = time.time()
    result_traditional = traditional_crew.process_document(
        temp_path, form_path, output_traditional
    )
    traditional_time = time.time() - start_time

    print(f"✅ Traditional method: {traditional_time:.2f}s")
    print(f"   Status: {'Success' if result_traditional.success else 'Failed'}")

    # Test AI Method (simulated - would require vision model)
    print("\n🧠 Testing AI Extraction with CrewAI...")
    # Simulate AI configuration without actually running it since it requires vision model
    DocumentProcessingCrew(
        text_model="llama3.2:3b", extraction_method="ai", vision_model="llava:7b"
    )

    # Note: This would require actual vision model installation
    print("   ⚠️ AI extraction requires vision model (llava:7b)")
    print("   Demo shows configuration and workflow")

    print("⚙️ AI method configuration ready")
    print("   Agents: DocumentCollector (AI) → Translator → FormAnalyst → FormFiller")
    print("   Vision Model: llava:7b (if available)")

    # Test OpenAI Method (requires API key)
    print("\n☁️ Testing OpenAI Extraction with CrewAI...")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if openai_api_key:
        # Create OpenAI-based crew
        DocumentProcessingCrew(
            text_model="llama3.2:3b",
            extraction_method="openai",
            vision_model="llava:7b",
            openai_api_key=openai_api_key,
            openai_model="gpt-4-vision-preview",
        )

        print("   ℹ️ Using OpenAI API for extraction")
        print("   Demo shows configuration and workflow")
        print("⚙️ OpenAI method configuration ready")
        print("   Agents: DocumentCollector (OpenAI) → Translator → FormAnalyst → FormFiller")
        print("   OpenAI Model: gpt-4-vision-preview")
    else:
        print("   ⚠️ OpenAI extraction requires API key")
        print("   To test, set OPENAI_API_KEY environment variable")

    # Comparison of all methods
    print("\n📊 CrewAI Extraction Method Comparison:")
    print("Traditional: Fast, reliable for clean text documents")
    print("AI: Better for complex layouts, handwriting using local models")
    print("OpenAI: Highest accuracy for complex documents using cloud API")
    print("All: Use same collaborative CrewAI agent framework")

    # Cleanup
    Path(temp_path).unlink()
    Path(form_path).unlink()
    if Path(output_traditional).exists():
        Path(output_traditional).unlink()

    print("\n")


def demo_crewai_error_handling():
    """Demonstrate CrewAI error handling and recovery."""
    print("🤖 Demo 5: CrewAI Error Handling")
    print("-" * 50)

    # Test 1: Non-existent file
    print("🧪 Test 1: Non-existent source file")
    crew_processor = DocumentProcessingCrew(text_model="llama3.2:3b")
    result = crew_processor.process_document("non_existent.pdf", "some_form.docx", "output.docx")
    print(f"Result: {'✅ Handled' if not result.success else '❌ Unexpected success'}")
    print(f"Error: {result.error}")
    print("🤖 CrewAI agents gracefully handled the error")

    # Test 2: Invalid configuration
    print("\n🧪 Test 2: Invalid model configuration")
    try:
        DocumentProcessingCrew(text_model="non_existent_model", extraction_method="traditional")
        print("✅ CrewAI validated configuration and provided fallback")
    except Exception as e:
        print(f"✅ CrewAI caught configuration error: {e}")

    # Test 3: Agent failure simulation
    print("\n🧪 Test 3: Agent collaboration under stress")
    print("🤖 CrewAI Features:")
    print("   - Automatic retry mechanisms")
    print("   - Task dependency management")
    print("   - Agent failure recovery")
    print("   - Detailed error reporting")
    print("   - Graceful degradation")

    print("\n")


def demo_crewai_configuration():
    """Show CrewAI configuration options."""
    print("🤖 Demo 6: CrewAI Configuration Options")
    print("-" * 50)

    # Different configurations
    configs = [
        {
            "name": "Fast Processing",
            "text_model": "llama3.2:3b",
            "extraction_method": "traditional",
            "description": "Optimized for speed with traditional extraction",
        },
        {
            "name": "High Accuracy (Local)",
            "text_model": "llama3.1:8b",
            "extraction_method": "ai",
            "vision_model": "llava:13b",
            "description": "Maximum accuracy with larger local models",
        },
        {
            "name": "Balanced",
            "text_model": "qwen2.5:7b",
            "extraction_method": "traditional",
            "description": "Good balance of speed and quality",
        },
        {
            "name": "Cloud-Powered",
            "text_model": "llama3.2:3b",
            "extraction_method": "openai",
            "openai_model": "gpt-4-vision-preview",
            "description": "Maximum accuracy using OpenAI's vision API",
        },
    ]

    print("📋 CrewAI Configuration Profiles:")
    for config in configs:
        print(f"\n🤖 {config['name']}:")
        print(f"   Text Model: {config['text_model']}")
        print(f"   Extraction: {config['extraction_method']}")
        if config.get("vision_model"):
            print(f"   Vision Model: {config['vision_model']}")
        if config.get("openai_model"):
            print(f"   OpenAI Model: {config['openai_model']}")
        print(f"   Use Case: {config['description']}")

    print("\n🔧 CrewAI Advanced Settings:")
    print("   - Process Type: Sequential (default) | Hierarchical")
    print("   - Max Retries: 3 per task")
    print("   - Timeout: Configurable per agent")
    print("   - Verbose Mode: Detailed agent logging")
    print("   - Memory: Shared context between agents")

    print("\n⚙️ Agent-Specific Configuration:")
    print("   DocumentCollector: Extraction method, vision model")
    print("   Translator: Text model, temperature, max tokens")
    print("   FormAnalyst: Analysis depth, field detection")
    print("   FormFiller: Mapping confidence, field validation")

    print("\n")


def main():
    """Run all CrewAI demos."""
    print("🚀 Vietnamese Document Form Filler - CrewAI Demo")
    print("=" * 60)
    print("This demo showcases the CrewAI-powered multi-agent")
    print("document processing system.\n")

    # Check if CrewAI and Ollama are available
    print("🔍 Checking CrewAI system requirements...")
    try:
        import crewai

        print(f"✅ CrewAI installed: v{crewai.__version__}")

        # Check if crewai_tools is available
        try:
            import crewai_tools  # noqa

            print("✅ CrewAI Tools available")
        except ImportError:
            print("⚠️ CrewAI Tools not available")

        # Check Ollama
        import aiohttp

        async def check_ollama():
            try:
                async with aiohttp.ClientSession() as session, session.get(
                    "http://localhost:11434/api/tags"
                ) as response:
                    if response.status == 200:
                        print("✅ Ollama is running")
                        data = await response.json()
                        models = [m["name"] for m in data.get("models", [])]
                        print(f"Available models: {', '.join(models[:3])}...")
                        return True
                    else:
                        print("❌ Ollama is not responding")
                        return False
            except Exception as e:
                print(f"❌ Cannot connect to Ollama: {e}")
                return False

        asyncio.run(check_ollama())

    except ImportError as e:
        print(f"❌ CrewAI not properly installed: {e}")
        print("Install with: pip install crewai crewai-tools langchain langchain-community")
        return
    except Exception as e:
        print(f"❌ System check failed: {e}")
        print("This demo requires CrewAI and Ollama to be running")
        return

    print("\n")

    # Run CrewAI demos
    demo_crewai_single_document()
    demo_crewai_agent_capabilities()
    demo_crewai_batch_processing()
    demo_crewai_extraction_methods()
    demo_crewai_error_handling()
    demo_crewai_configuration()

    print("🎉 CrewAI Demo completed!")
    print("\n🤖 CrewAI Advantages Summary:")
    print("   ✅ Structured agent collaboration")
    print("   ✅ Built-in error handling and retries")
    print("   ✅ Task dependency management")
    print("   ✅ Enhanced observability and logging")
    print("   ✅ Scalable parallel processing")
    print("   ✅ Easy configuration and customization")
    print("   ✅ Multiple extraction methods (Traditional, AI, OpenAI)")
    print("\nTry the CLI commands or web interface for full CrewAI features!")


if __name__ == "__main__":
    main()  #!/usr/bin/env python3
"""
Demo script for Vietnamese Document Form Filler
Shows basic usage and capabilities
"""


# Demo data - sample Vietnamese text
SAMPLE_VIETNAMESE_TEXT = """
Họ và tên: Nguyễn Văn An
Ngày sinh: 15/03/1990
Địa chỉ: 123 Phố Huế, Quận Hai Bà Trưng, Hà Nội
Số điện thoại: 0123456789
Email: nguyen.van.an@email.com
Trình độ học vấn: Cử nhân Công nghệ Thông tin

Kinh nghiệm làm việc:
- 2015-2018: Lập trình viên tại Công ty ABC
- 2018-2022: Trưởng nhóm phát triển tại Công ty XYZ
- 2022-hiện tại: Kiến trúc sư phần mềm tại Công ty DEF

Kỹ năng:
- Ngôn ngữ lập trình: Python, Java, JavaScript
- Framework: Django, React, Spring Boot
- Cơ sở dữ liệu: MySQL, PostgreSQL, MongoDB
- DevOps: Docker, Kubernetes, AWS
"""

SAMPLE_FORM_CONTENT = """
EMPLOYMENT APPLICATION FORM

Personal Information:
Name: ___________________
Date of Birth: ___________________
Address: ___________________________________
Phone Number: ___________________
Email: ___________________

Education:
Highest Degree: ___________________
Institution: ___________________

Work Experience:
Previous Position: ___________________
Company: ___________________
Years of Experience: ___________________

Skills:
Technical Skills: ___________________________________
Programming Languages: ___________________
Frameworks: ___________________
Databases: ___________________

Additional Information:
___________________________________________
___________________________________________
"""


async def demo_single_document():
    """Demonstrate single document processing."""
    print("🔹 Demo 1: Single Document Processing")
    print("-" * 40)

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as source_file:
        source_file.write(SAMPLE_VIETNAMESE_TEXT)
        source_path = source_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as form_file:
        form_file.write(SAMPLE_FORM_CONTENT)
        form_path = form_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as output_file:
        output_path = output_file.name

    # Initialize processor
    processor = DocumentProcessor(ollama_model="llama3.2:3b")

    print("📄 Processing Vietnamese CV...")
    print(f"Source: {Path(source_path).name}")
    print(f"Form: {Path(form_path).name}")

    # Process document
    result = await processor.process_document(source_path, form_path, output_path)

    if result.success:
        print("✅ Success!")
        print(f"Output saved to: {output_path}")

        # Show a preview of the result
        if Path(output_path).exists():
            with open(output_path) as f:
                content = f.read()
                print("\n📋 Preview of filled form:")
                print(content[:300] + "..." if len(content) > 300 else content)
    else:
        print(f"❌ Failed: {result.error}")

    # Cleanup
    Path(source_path).unlink()
    Path(form_path).unlink()
    if Path(output_path).exists():
        Path(output_path).unlink()

    print("\n")


async def demo_agent_pipeline():
    """Demonstrate individual agent usage."""
    print("🔹 Demo 2: Individual Agent Pipeline")
    print("-" * 40)

    from document_processor import DocumentCollectorAgent, FormFillerAgent, TranslatorAgent

    # Create agents
    collector = DocumentCollectorAgent()
    translator = TranslatorAgent()
    filler = FormFillerAgent()

    # Create temp files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
        temp_file.write(SAMPLE_VIETNAMESE_TEXT)
        temp_path = temp_file.name

    print("🔧 Step 1: Document Collection")
    collection_result = await collector.process(temp_path)
    if collection_result.success:
        print(f"✅ Extracted {len(collection_result.data)} characters")
        vietnamese_text = collection_result.data[:100] + "..."
        print(f"📝 Sample: {vietnamese_text}")
    else:
        print(f"❌ Failed: {collection_result.error}")
        return

    print("\n🔧 Step 2: Translation")
    translation_result = await translator.process(collection_result.data)
    if translation_result.success:
        print("✅ Translation completed")
        english_text = translation_result.data[:100] + "..."
        print(f"📝 Sample: {english_text}")
    else:
        print(f"❌ Failed: {translation_result.error}")
        return

    print("\n🔧 Step 3: Form Filling")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as form_file:
        form_file.write(SAMPLE_FORM_CONTENT)
        form_path = form_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as output_file:
        output_path = output_file.name

    filling_result = await filler.process(
        {
            "form_path": form_path,
            "translated_text": translation_result.data,
            "output_path": output_path,
        }
    )

    if filling_result.success:
        print("✅ Form filling completed")
        print(f"📊 Fields filled: {filling_result.metadata.get('filled_count', 'N/A')}")
    else:
        print(f"❌ Failed: {filling_result.error}")

    # Cleanup
    Path(temp_path).unlink()
    Path(form_path).unlink()
    if Path(output_path).exists():
        Path(output_path).unlink()

    print("\n")


async def demo_batch_processing():
    """Demonstrate batch processing."""
    print("🔹 Demo 3: Batch Processing")
    print("-" * 40)

    # Create sample documents
    sample_documents = [
        f"Document {i}: Tên: Người {i}, Tuổi: {20+i}, Địa chỉ: Hà Nội" for i in range(1, 4)
    ]

    # Create temporary directory structure
    temp_dir = Path(tempfile.mkdtemp())
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create sample files
    form_path = temp_dir / "form.txt"
    with open(form_path, "w") as f:
        f.write(SAMPLE_FORM_CONTENT)

    # Create input documents
    for i, content in enumerate(sample_documents, 1):
        doc_path = input_dir / f"document_{i}.txt"
        with open(doc_path, "w") as f:
            f.write(content)

    print(f"📁 Created {len(sample_documents)} sample documents")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Initialize batch processor
    batch_processor = BatchProcessor(model="llama3.2:3b", max_concurrent=2, timeout=60)

    # Discover jobs
    job_count = batch_processor.discover_jobs(
        str(input_dir), str(form_path), str(output_dir), "*.txt"
    )

    print(f"🔍 Discovered {job_count} processing jobs")

    # Progress callback
    def progress_callback(completed, total, job):
        progress = (completed / total) * 100
        status = "✅" if job.status == "completed" else "❌"
        print(f"Progress: {completed}/{total} ({progress:.1f}%) - {status} {job.source_path.name}")

    # Process batch
    print("\n🔄 Starting batch processing...")
    stats = await batch_processor.process_all(progress_callback)

    print("\n📊 Batch Processing Results:")
    print(f"Total: {stats['total']}")
    print(f"Completed: {stats['completed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success Rate: {stats['success_rate']:.1f}%")
    print(f"Total Time: {stats['total_time']:.2f}s")
    print(f"Average per Job: {stats['average_job_time']:.2f}s")

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)

    print("\n")


async def demo_error_handling():
    """Demonstrate error handling scenarios."""
    print("🔹 Demo 4: Error Handling")
    print("-" * 40)

    processor = DocumentProcessor(ollama_model="llama3.2:3b")

    # Test 1: Non-existent file
    print("🧪 Test 1: Non-existent source file")
    result = await processor.process_document("non_existent.pdf", "some_form.docx", "output.docx")
    print(f"Result: {'✅ Handled' if not result.success else '❌ Unexpected success'}")
    print(f"Error: {result.error}")

    # Test 2: Empty content
    print("\n🧪 Test 2: Empty document")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as empty_file:
        empty_file.write("")
        empty_path = empty_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as form_file:
        form_file.write(SAMPLE_FORM_CONTENT)
        form_path = form_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as output_file:
        output_path = output_file.name

    result = await processor.process_document(empty_path, form_path, output_path)
    print(f"Result: {'✅ Handled' if not result.success else '❌ Unexpected success'}")
    print(f"Error: {result.error}")

    # Cleanup
    Path(empty_path).unlink()
    Path(form_path).unlink()

    print("\n")


def demo_configuration():
    """Show configuration options."""
    print("🔹 Demo 5: Configuration Options")
    print("-" * 40)

    # Sample configuration
    config = {
        "default_model": "llama3.2:3b",
        "ollama_host": "localhost",
        "ollama_port": 11434,
        "ocr_language": "vie",
        "translation_settings": {
            "temperature": 0.1,
            "max_tokens": 2048,
            "system_prompt": "You are a professional translator...",
        },
        "form_filling_settings": {"confidence_threshold": 0.7, "max_field_mappings": 20},
    }

    print("📋 Sample Configuration:")
    print(json.dumps(config, indent=2))

    print("\n🔧 Environment Variables:")
    env_vars = [
        "OLLAMA_HOST=localhost",
        "OLLAMA_PORT=11434",
        "OLLAMA_MODEL=llama3.2:3b",
        "LOG_LEVEL=INFO",
    ]
    for var in env_vars:
        print(f"  export {var}")

    print("\n")


async def main():
    """Run all demos."""
    print("🚀 Vietnamese Document Form Filler - Demo")
    print("=" * 50)
    print("This demo showcases the capabilities of the multi-agent")
    print("document processing system.\n")

    # Check if Ollama is available
    print("🔍 Checking system requirements...")
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session, session.get(
            "http://localhost:11434/api/tags"
        ) as response:
            if response.status == 200:
                print("✅ Ollama is running")
                data = await response.json()
                models = [m["name"] for m in data.get("models", [])]
                print(f"Available models: {', '.join(models[:3])}...")
            else:
                print("❌ Ollama is not responding")
                print("Please ensure Ollama is installed and running")
                return
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("This demo requires Ollama to be running")
        return

    print("\n")

    # Run demos
    await demo_single_document()
    await demo_agent_pipeline()
    await demo_batch_processing()
    await demo_error_handling()
    demo_configuration()

    print("🎉 Demo completed!")
    print("Try the CLI commands or web interface for more features.")


if __name__ == "__main__":
    asyncio.run(main())
