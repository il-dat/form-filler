import asyncio
import json
import tempfile
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from form_filler.batch_processor import CrewAIBatchProcessor

# Import CrewAI components
from form_filler.crew import DocumentProcessingCrew
from form_filler.models import ProcessingResult

import streamlit as st

# Set page config
st.set_page_config(
    page_title="Vietnamese Document Form Filler (CrewAI)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-processing {
        color: #ffc107;
        font-weight: bold;
    }
    .crew-badge {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "processing_history" not in st.session_state:
    st.session_state.processing_history = []

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

# Header
st.markdown('<h1 class="main-header">Vietnamese Document Form Filler</h1>', unsafe_allow_html=True)
st.markdown(
    "### 🤖 **CrewAI-Powered Multi-Agent System** for processing Vietnamese documents and filling English DOCX forms"
)

# Sidebar
st.sidebar.title("CrewAI Settings")

# Model selection
st.sidebar.subheader("🤖 AI Models")
available_text_models = ["llama3.2:3b", "llama3.1:8b", "qwen2.5:7b", "phi3:3.8b"]
selected_text_model = st.sidebar.selectbox(
    "Text Model (Translation & Form Filling)", available_text_models
)

# Extraction method selection
st.sidebar.subheader("📄 Text Extraction")
extraction_method = st.sidebar.radio(
    "Extraction Method",
    ["traditional", "ai"],
    help="Traditional: PyMuPDF + Tesseract | AI: Vision models with CrewAI agents",
)

if extraction_method == "ai":
    vision_models = ["llava:7b", "llava:13b", "bakllava", "llava-phi3"]
    vision_model = st.sidebar.selectbox(
        "Vision Model", vision_models, help="Select vision model for AI-powered text extraction"
    )

    st.sidebar.info(
        """
    🤖 **CrewAI AI Extraction:**
    - Uses specialized vision agent
    - Better context understanding
    - Improved accuracy for complex layouts
    - Collaborative multi-agent approach
    """
    )
else:
    vision_model = "llava:7b"  # Default, won't be used

# Processing options
st.sidebar.subheader("⚙️ Processing Options")
max_concurrent = st.sidebar.slider(
    "Max Concurrent Crews", 1, 8, 3, help="Number of parallel CrewAI teams"
)
timeout = st.sidebar.slider("Timeout per Crew (seconds)", 30, 900, 300)

# CrewAI Status
st.sidebar.subheader("🔍 System Status")
if st.sidebar.button("Check CrewAI Status"):
    check_crewai_status()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Single Document", "Batch Processing", "History", "System Status"]
)

# Tab 1: Single Document Processing
with tab1:
    st.header("🤖 CrewAI Single Document Processing")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Vietnamese Document")
        source_file = st.file_uploader(
            "Choose a Vietnamese document",
            type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff"],
            help="Upload a PDF or image file containing Vietnamese text",
        )

        if source_file:
            st.success(f"✅ Uploaded: {source_file.name}")
            st.info(f"File size: {source_file.size / 1024:.2f} KB")

    with col2:
        st.subheader("Upload DOCX Form Template")
        form_file = st.file_uploader(
            "Choose a DOCX form template",
            type=["docx"],
            help="Upload the English DOCX form to be filled",
        )

        if form_file:
            st.success(f"✅ Uploaded: {form_file.name}")
            st.info(f"File size: {form_file.size / 1024:.2f} KB")

    # Processing section
    if source_file and form_file:
        st.subheader("CrewAI Processing Pipeline")

        # Show CrewAI workflow
        with st.expander("🔍 View CrewAI Workflow", expanded=False):
            st.markdown(
                """
            **CrewAI Multi-Agent Pipeline:**
            1. 📄 **Document Collector Agent** - Extracts text from Vietnamese documents
            2. 🔄 **Translator Agent** - Translates Vietnamese to English
            3. 🔍 **Form Analyst Agent** - Analyzes DOCX form structure
            4. ✍️ **Form Filler Agent** - Intelligently fills form fields

            Each agent is specialized and works collaboratively to ensure optimal results.
            """
            )

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("🚀 Start CrewAI Processing", type="primary"):
                process_single_document_crew(
                    source_file, form_file, selected_text_model, extraction_method, vision_model
                )

        with col2:
            if st.button("👁️ Preview Source"):
                preview_document(source_file)

        with col3:
            output_filename = st.text_input(
                "Output filename", value=f"{Path(source_file.name).stem}_filled.docx"
            )

        # Show configuration
        st.info(
            f"🤖 **CrewAI Config**: {extraction_method} extraction | Text: {selected_text_model}"
            + (f" | Vision: {vision_model}" if extraction_method == "ai" else "")
        )

# Tab 2: Batch Processing
with tab2:
    st.header("🤖 CrewAI Batch Processing")

    batch_mode = st.radio(
        "Choose batch processing mode:",
        ["Upload Multiple Files", "Upload Folder (ZIP)", "Generate from Directory"],
    )

    if batch_mode == "Upload Multiple Files":
        st.subheader("Upload Multiple Vietnamese Documents")
        source_files = st.file_uploader(
            "Choose Vietnamese documents",
            type=["pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Upload multiple documents to process with CrewAI",
        )

        form_template = st.file_uploader(
            "Choose DOCX form template", type=["docx"], key="batch_form"
        )

        if source_files and form_template:
            st.success(f"✅ Ready to process {len(source_files)} documents with CrewAI")

            # Show CrewAI batch configuration
            with st.expander("🔧 CrewAI Batch Configuration", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents", len(source_files))
                with col2:
                    st.metric("Concurrent Crews", max_concurrent)
                with col3:
                    estimated_time = len(source_files) * 90 / max_concurrent
                    st.metric("Est. Time", f"{estimated_time/60:.1f} min")

            if st.button("🚀 Start CrewAI Batch Processing", type="primary"):
                process_batch_documents_crew(
                    source_files,
                    form_template,
                    selected_text_model,
                    max_concurrent,
                    timeout,
                    extraction_method,
                    vision_model,
                )

    elif batch_mode == "Upload Folder (ZIP)":
        st.subheader("Upload ZIP Archive")
        zip_file = st.file_uploader(
            "Choose a ZIP file containing documents",
            type=["zip"],
            help="Upload a ZIP archive with Vietnamese documents for CrewAI processing",
        )

        form_template = st.file_uploader(
            "Choose DOCX form template", type=["docx"], key="batch_form_zip"
        )

        if zip_file and form_template:
            if st.button("📦 Extract and Process with CrewAI", type="primary"):
                process_zip_archive_crew(
                    zip_file,
                    form_template,
                    selected_text_model,
                    max_concurrent,
                    timeout,
                    extraction_method,
                    vision_model,
                )

# Tab 3: History
with tab3:
    st.header("📊 Processing History")

    if st.session_state.processing_history:
        # Create DataFrame from history
        df = pd.DataFrame(st.session_state.processing_history)

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Processed", len(df))
        with col2:
            success_count = len(df[df["status"] == "success"])
            st.metric("Successful", success_count)
        with col3:
            failed_count = len(df[df["status"] == "failed"])
            st.metric("Failed", failed_count)
        with col4:
            avg_time = df["processing_time"].mean()
            st.metric("Avg. Time (s)", f"{avg_time:.2f}")

        # CrewAI specific metrics
        if "extraction_method" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                extraction_counts = df["extraction_method"].value_counts()
                fig_extraction = px.pie(
                    values=extraction_counts.values,
                    names=extraction_counts.index,
                    title="Extraction Methods Used",
                )
                st.plotly_chart(fig_extraction, use_container_width=True)
            with col2:
                # Processing time by extraction method
                fig_time = px.box(
                    df,
                    x="extraction_method",
                    y="processing_time",
                    title="Processing Time by Extraction Method",
                )
                st.plotly_chart(fig_time, use_container_width=True)

        # Processing time chart
        fig = px.bar(
            df,
            x="filename",
            y="processing_time",
            color="status",
            title="CrewAI Processing Time by Document",
            hover_data=["extraction_method"] if "extraction_method" in df.columns else None,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed history table
        st.subheader("Detailed CrewAI Processing History")
        display_columns = ["timestamp", "filename", "status", "processing_time"]
        if "extraction_method" in df.columns:
            display_columns.append("extraction_method")
        if "crew_id" in df.columns:
            display_columns.append("crew_id")
        display_columns.append("error")

        st.dataframe(df[display_columns])

        # Download history
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CrewAI History as CSV",
            data=csv,
            file_name=f"crewai_processing_history_{int(time.time())}.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "No CrewAI processing history available yet. Process some documents to see statistics!"
        )

# Tab 4: System Status
with tab4:
    st.header("🔍 System Status")

    # CrewAI Status Section
    st.subheader("🤖 CrewAI System Status")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Check CrewAI Status"):
            check_crewai_status()

    with col2:
        if st.button("🔧 Check CrewAI Tools"):
            check_crewai_tools()

    # Ollama Models Section
    st.subheader("🧠 Ollama Models")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📋 List Text Models"):
            list_text_models()

    with col2:
        if st.button("👁️ List Vision Models"):
            list_vision_models()

    # System Resources
    st.subheader("💻 System Resources")
    display_system_metrics()

    # Current Configuration
    st.subheader("⚙️ Current CrewAI Configuration")
    config_data = {
        "Text Model": selected_text_model,
        "Extraction Method": extraction_method,
        "Vision Model": vision_model if extraction_method == "ai" else "N/A",
        "Max Concurrent Crews": max_concurrent,
        "Timeout per Crew (seconds)": timeout,
        "Ollama Host": "localhost",
        "Ollama Port": 11434,
        "CrewAI Process": "Sequential",
        "Agent Types": ["DocumentCollector", "Translator", "FormAnalyst", "FormFiller"],
    }

    # Display config in two columns
    col1, col2 = st.columns(2)
    with col1:
        for key, value in list(config_data.items())[: len(config_data) // 2]:
            st.text(f"{key}: {value}")
    with col2:
        for key, value in list(config_data.items())[len(config_data) // 2 :]:
            st.text(f"{key}: {value}")

    # CrewAI Agent Architecture
    with st.expander("🏗️ CrewAI Agent Architecture", expanded=False):
        st.markdown(
            """
        **Multi-Agent Workflow:**

        1. **📄 Document Collector Agent**
           - Role: Extract text from Vietnamese documents
           - Tools: DocumentExtractionTool (traditional/AI)
           - Capabilities: PDF processing, OCR, Vision models

        2. **🔄 Translator Agent**
           - Role: Vietnamese to English translation
           - Tools: TranslationTool
           - LLM: Selected text model

        3. **🔍 Form Analyst Agent**
           - Role: Analyze DOCX form structure
           - Tools: FormAnalysisTool
           - Capabilities: Field detection, context understanding

        4. **✍️ Form Filler Agent**
           - Role: Intelligent form completion
           - Tools: FormFillingTool
           - Features: Smart field mapping, AI-assisted content placement
        """
        )


# Helper functions (CrewAI versions)
def process_single_document_crew(
    source_file, form_file, text_model, extraction_method, vision_model
):
    """Process a single document using CrewAI"""
    with st.spinner("🤖 CrewAI agents are working..."):
        # Show progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()

        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(source_file.name).suffix
        ) as tmp_source:
            tmp_source.write(source_file.getvalue())
            tmp_source_path = tmp_source.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_form:
            tmp_form.write(form_file.getvalue())
            tmp_form_path = tmp_form.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_output:
            tmp_output_path = tmp_output.name

        # Create CrewAI processor
        status_text.text("🤖 Initializing CrewAI agents...")
        progress_bar.progress(10)

        processor = DocumentProcessingCrew(
            text_model=text_model, extraction_method=extraction_method, vision_model=vision_model
        )

        status_text.text("📄 Document Collector Agent extracting text...")
        progress_bar.progress(25)

        # Process document (this is synchronous)
        result = processor.process_document(tmp_source_path, tmp_form_path, tmp_output_path)

        progress_bar.progress(100)
        processing_time = time.time() - start_time

        # Clean up temp files
        Path(tmp_source_path).unlink()
        Path(tmp_form_path).unlink()

        # Handle result
        if result.success:
            st.success(f"✅ CrewAI successfully processed document in {processing_time:.2f}s")

            # Show which agents were involved
            agent_info = f"🤖 **Agents used**: Document Collector ({extraction_method}) → Translator → Form Analyst → Form Filler"
            st.info(agent_info)

            # Download button for filled form
            with open(tmp_output_path, "rb") as f:
                st.download_button(
                    label="📥 Download CrewAI Filled Form",
                    data=f.read(),
                    file_name=f"{Path(source_file.name).stem}_crewai_filled.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )

            # Add to history with CrewAI-specific metadata
            st.session_state.processing_history.append(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": source_file.name,
                    "status": "success",
                    "processing_time": processing_time,
                    "extraction_method": extraction_method,
                    "text_model": text_model,
                    "vision_model": vision_model if extraction_method == "ai" else None,
                    "agent_type": "CrewAI",
                    "error": None,
                }
            )
        else:
            st.error(f"❌ CrewAI processing failed: {result.error}")

            # Add to history
            st.session_state.processing_history.append(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": source_file.name,
                    "status": "failed",
                    "processing_time": processing_time,
                    "extraction_method": extraction_method,
                    "text_model": text_model,
                    "vision_model": vision_model if extraction_method == "ai" else None,
                    "agent_type": "CrewAI",
                    "error": result.error,
                }
            )

        # Clean up output file
        if Path(tmp_output_path).exists():
            Path(tmp_output_path).unlink()


def process_batch_documents_crew(
    source_files,
    form_template,
    text_model,
    max_concurrent,
    timeout,
    extraction_method,
    vision_model,
):
    """Process multiple documents in batch using CrewAI"""
    st.subheader("🤖 CrewAI Batch Processing Progress")

    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.empty()
    crew_status_container = st.empty()

    # Initialize CrewAI batch processor
    batch_processor = CrewAIBatchProcessor(
        text_model=text_model,
        extraction_method=extraction_method,
        vision_model=vision_model,
        max_concurrent=max_concurrent,
        timeout=timeout,
    )

    # Save form template
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_form:
        tmp_form.write(form_template.getvalue())
        form_path = tmp_form.name

    # Add jobs
    temp_files = []
    for source_file in source_files:
        # Save source file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(source_file.name).suffix
        ) as tmp_source:
            tmp_source.write(source_file.getvalue())
            source_path = tmp_source.name
            temp_files.append(source_path)

        # Create output path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_output:
            output_path = tmp_output.name

        batch_processor.add_job(source_path, form_path, output_path)

    # Progress callback
    results = []
    active_crews = set()

    def progress_callback(completed, total, job):
        progress = completed / total
        progress_bar.progress(progress)
        status_text.text(f"🤖 CrewAI Processing: {completed}/{total} ({progress*100:.1f}%)")

        # Track active crews
        if job.status == "processing":
            active_crews.add(job.crew_id)
        elif job.status in ["completed", "failed"]:
            active_crews.discard(job.crew_id)

        # Update crew status
        crew_status_container.info(
            f"Active Crews: {', '.join(sorted(active_crews)) if active_crews else 'None'}"
        )

        # Add result to list
        results.append(
            {
                "filename": Path(job.source_path).name,
                "status": job.status,
                "processing_time": job.end_time - job.start_time if job.end_time > 0 else 0,
                "crew_id": job.crew_id,
                "error": job.error,
                "extraction_method": extraction_method,
            }
        )

        # Update results display
        with results_container.container():
            df = pd.DataFrame(results)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Completed", completed)
            with col2:
                success_count = len([r for r in results if r["status"] == "completed"])
                st.metric("Successful", success_count)
            with col3:
                failed_count = len([r for r in results if r["status"] == "failed"])
                st.metric("Failed", failed_count)
            with col4:
                avg_time = df[df["status"] == "completed"]["processing_time"].mean()
                st.metric("Avg Time", f"{avg_time:.1f}s" if not pd.isna(avg_time) else "N/A")

            # Show crew performance
            if len(df) > 0:
                fig = px.scatter(
                    df,
                    x="processing_time",
                    y="crew_id",
                    color="status",
                    title="CrewAI Processing Performance by Crew",
                )
                st.plotly_chart(fig, use_container_width=True)

    # Process all jobs
    with st.spinner("🤖 CrewAI teams are working in parallel..."):
        batch_stats = batch_processor.process_all(progress_callback)

    # Clean up temp files
    Path(form_path).unlink()
    for temp_file in temp_files:
        if Path(temp_file).exists():
            Path(temp_file).unlink()

    # Display final results
    st.success("✅ CrewAI batch processing completed!")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Jobs", batch_stats["total"])
    with col2:
        st.metric("Completed", batch_stats["completed"])
    with col3:
        st.metric("Success Rate", f"{batch_stats['success_rate']:.1f}%")
    with col4:
        st.metric("Total Time", f"{batch_stats['total_time']:.2f}s")

    # Show CrewAI specific metrics
    st.subheader("🤖 CrewAI Performance Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Extraction Method", batch_stats["extraction_method"])
    with col2:
        st.metric("Text Model", batch_stats["text_model"])
    with col3:
        st.metric("Max Concurrent Crews", max_concurrent)

    # Store results
    st.session_state.batch_results = batch_stats

    # Download option for successful results
    successful_jobs = [job for job in batch_processor.jobs if job.status == "completed"]
    if successful_jobs:
        st.subheader("📥 Download CrewAI Results")

        # Create ZIP file with all successful results
        import io
        import zipfile

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for job in successful_jobs:
                if Path(job.output_path).exists():
                    zip_file.write(
                        job.output_path, f"{Path(job.source_path).stem}_crewai_filled.docx"
                    )

        st.download_button(
            label="📦 Download All CrewAI Filled Forms (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"crewai_batch_results_{int(time.time())}.zip",
            mime="application/zip",
        )



def check_crewai_status():
    """Check CrewAI installation and status"""
    try:
        import crewai

        st.success(f"✅ CrewAI installed: v{crewai.__version__}")

        # Check available tools
        try:
            from crewai_tools import tool

            st.success("✅ CrewAI Tools available")
        except ImportError:
            st.warning("⚠️ CrewAI Tools not found. Install with: pip install crewai-tools")

        # Check Langchain integration
        try:
            from langchain_ollama import OllamaLLM

            st.success("✅ Langchain-Ollama integration ready")
        except ImportError:
            st.error("❌ Langchain-Ollama not available")

    except ImportError:
        st.error("❌ CrewAI not installed. Run: pip install crewai crewai-tools")


def check_crewai_tools():
    """Check available CrewAI tools"""
    from form_filler.tools import (
        DocumentExtractionTool,
        FormAnalysisTool,
        FormFillingTool,
        TranslationTool,
    )

    tools_status = {
        "DocumentExtractionTool": "✅ Ready",
        "TranslationTool": "✅ Ready",
        "FormAnalysisTool": "✅ Ready",
        "FormFillingTool": "✅ Ready",
    }

    for tool_name, status in tools_status.items():
        st.text(f"{tool_name}: {status}")


def list_text_models():
    """List available text models"""
    import aiohttp

    async def get_models():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])

                        text_models = []
                        for model in models:
                            name = model.get("name", "Unknown")
                            if not any(v in name.lower() for v in ["llava", "vision", "bakllava"]):
                                size = model.get("size", 0)
                                size_mb = size / (1024 * 1024) if size else 0
                                text_models.append(
                                    {
                                        "Model": name,
                                        "Size (MB)": f"{size_mb:.1f}",
                                        "Type": "Text/Translation",
                                    }
                                )

                        if text_models:
                            df = pd.DataFrame(text_models)
                            st.dataframe(df)
                        else:
                            st.warning("No text models found")
                    else:
                        st.error("Failed to fetch models")
        except Exception as e:
            st.error(f"Error: {e}")

    asyncio.run(get_models())


def list_vision_models():
    """List available vision models"""
    import aiohttp

    async def get_models():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])

                        vision_models = []
                        for model in models:
                            name = model.get("name", "Unknown")
                            if any(v in name.lower() for v in ["llava", "vision", "bakllava"]):
                                size = model.get("size", 0)
                                size_mb = size / (1024 * 1024) if size else 0
                                vision_models.append(
                                    {
                                        "Model": name,
                                        "Size (MB)": f"{size_mb:.1f}",
                                        "Type": "Vision/OCR",
                                    }
                                )

                        if vision_models:
                            df = pd.DataFrame(vision_models)
                            st.dataframe(df)
                            st.info(
                                "💡 These models enable AI-powered text extraction from images and PDFs"
                            )
                        else:
                            st.warning(
                                "No vision models found. Install with: ollama pull llava:7b"
                            )
                    else:
                        st.error("Failed to fetch models")
        except Exception as e:
            st.error(f"Error: {e}")

    asyncio.run(get_models())


def preview_document(source_file):
    """Preview the source document"""
    if source_file.type == "application/pdf":
        st.info("PDF preview not implemented in this demo")
    else:
        from PIL import Image

        image = Image.open(source_file)
        st.image(image, caption=source_file.name, use_column_width=True)


def check_ollama_status():
    """Check if Ollama is running"""
    import requests

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            st.success("✅ Ollama is running")
            data = response.json()
            models = data.get("models", [])
            st.info(f"Available models: {len(models)}")
        else:
            st.error(f"❌ Ollama responded with status: {response.status_code}")
    except requests.RequestException as e:
        st.error(f"❌ Could not connect to Ollama: {e}")


def display_system_metrics():
    """Display basic system metrics"""
    try:
        import psutil

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        st.metric("CPU Usage", f"{cpu_percent}%")

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        st.metric("Memory Usage", f"{memory_percent}%")

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100
        st.metric("Disk Usage", f"{disk_percent:.1f}%")
    except ImportError:
        st.info("Install psutil for system metrics: pip install psutil")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Vietnamese Document Form Filler v2.0 (CrewAI Edition) | Built with ❤️ using CrewAI & Streamlit</p>
        <p>🤖 Powered by autonomous AI agents working in harmony</p>
    </div>
    """,
    unsafe_allow_html=True,
)
