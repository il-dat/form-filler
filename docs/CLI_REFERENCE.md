# Form Filler CLI Reference

This document provides a complete reference for using the Form Filler command line interface (CLI).

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Global Options](#global-options)
- [Commands](#commands)
  - [process](#process)
  - [extract](#extract)
  - [translate](#translate)
  - [analyze](#analyze)
  - [fill](#fill)
- [Batch Processing](#batch-processing)
- [Environment Variables](#environment-variables)
- [Return Codes](#return-codes)

## Installation

```bash
pip install form-filler
```

## Basic Usage

```bash
form-filler [COMMAND] [OPTIONS]
```

## Global Options

| Option          | Description                                      |
|-----------------|--------------------------------------------------|
| `--version`     | Display version information and exit.            |
| `--help`        | Display help text for the command.               |
| `--verbose`     | Enable verbose output for detailed logging.      |

All commands display a progress bar or spinner while they are executing, providing visual feedback during long-running operations.

## AI Provider Options

These options control which AI provider is used for operations that require AI capabilities:

| Option           | Description                                       |
|------------------|---------------------------------------------------|
| `--provider`     | AI provider to use (default: "ollama").           |
| `--api-key`      | API key for the AI provider (if needed).          |
| `--api-base`     | Base URL for the AI provider API (if custom).     |
| `--model`        | Model name to use with the selected provider.     |

Available providers:
- `ollama`: Local models via Ollama
- `openai`: OpenAI models (GPT-3.5, GPT-4, etc.)
- `anthropic`: Anthropic Claude models
- `deepseek`: DeepSeek AI models
- `gemini`: Google Gemini models

## Commands

### `process`

Process a document completely - extract text, translate if needed, and fill a form.

```bash
form-filler process [OPTIONS] [INPUT_FILE] [FORM_FILE] [OUTPUT_FILE]
```

#### Options:

| Option           | Description                                            |
|------------------|--------------------------------------------------------|
| `--model`        | LLM model to use (default: "llama3.2:3b").             |
| `--provider`     | AI provider to use (default: "ollama").                |
| `--api-key`      | API key for the AI provider (if needed).               |
| `--api-base`     | Base URL for the AI provider API (if custom).          |
| `--translate`    | Translate the input document if it's in Vietnamese.    |

#### Examples:

```bash
# Process a document using default Ollama model
form-filler process scan.pdf form.docx output.docx

# Process with translation
form-filler process vietnamese_doc.png form.docx output.docx --translate

# Process using OpenAI models
form-filler process scan.pdf form.docx output.docx --provider openai --api-key your_api_key --model gpt-4

# Process using Anthropic Claude
form-filler process scan.pdf form.docx output.docx --provider anthropic --api-key your_api_key --model claude-3-opus-20240229

# Process using Google Gemini
form-filler process scan.pdf form.docx output.docx --provider gemini --api-key your_api_key --model gemini-1.5-pro
```

### `extract`

Extract text from a document (PDF or image).

```bash
form-filler extract [OPTIONS] [INPUT_FILE]
```

#### Options:

| Option           | Description                                            |
|------------------|--------------------------------------------------------|
| `--model`        | LLM model to use (default: "llama3.2:3b").             |
| `--provider`     | AI provider to use (default: "ollama").                |
| `--api-key`      | API key for the AI provider (if needed).               |
| `--api-base`     | Base URL for the AI provider API (if custom).          |
| `--method`       | Extraction method: "traditional", "ai", or "auto" (default).  |
| `--output`       | Output file to save the extracted text (optional).     |

#### Examples:

```bash
# Extract text from a PDF
form-filler extract document.pdf

# Extract with specific method and save to file
form-filler extract receipt.png --method ai --output extracted.txt

# Extract using OpenAI
form-filler extract document.pdf --provider openai --api-key your_api_key --model gpt-4-vision-preview

# Extract using DeepSeek
form-filler extract document.pdf --provider deepseek --api-key your_api_key --model deepseek-vision-large
```

### `translate`

Translate text from Vietnamese to English.

```bash
form-filler translate [OPTIONS] [INPUT_TEXT_OR_FILE]
```

#### Options:

| Option           | Description                                            |
|------------------|--------------------------------------------------------|
| `--model`        | LLM model to use (default: "llama3.2:3b").             |
| `--provider`     | AI provider to use (default: "ollama").                |
| `--api-key`      | API key for the AI provider (if needed).               |
| `--api-base`     | Base URL for the AI provider API (if custom).          |
| `--output`       | Output file to save the translated text (optional).    |
| `--file`         | Treat input as a file path instead of raw text.        |

#### Examples:

```bash
# Translate a piece of text
form-filler translate "Xin chào từ Việt Nam"

# Translate content from a file
form-filler translate vietnamese_text.txt --file

# Translate to a specific file
form-filler translate input.txt --file --output translated.txt

# Translate using a specific provider
form-filler translate vietnamese_text.txt --file --provider openai --api-key your_api_key --model gpt-4
```

### `analyze`

Analyze a form to identify fillable fields.

```bash
form-filler analyze [OPTIONS] [FORM_FILE]
```

#### Options:

| Option           | Description                                            |
|------------------|--------------------------------------------------------|
| `--output`       | Output file to save the analysis results (optional).   |

#### Examples:

```bash
# Analyze a form
form-filler analyze form.docx

# Save analysis results
form-filler analyze form.docx --output analysis.json
```

### `fill`

Fill a form with provided content.

```bash
form-filler fill [OPTIONS] [FORM_FILE] [CONTENT] [OUTPUT_FILE]
```

#### Options:

| Option           | Description                                            |
|------------------|--------------------------------------------------------|
| `--model`        | LLM model to use (default: "llama3.2:3b").             |
| `--provider`     | AI provider to use (default: "ollama").                |
| `--api-key`      | API key for the AI provider (if needed).               |
| `--api-base`     | Base URL for the AI provider API (if custom).          |
| `--mappings`     | JSON file with field mappings (optional).              |
| `--from-file`    | Read content from file instead of command line.        |

#### Examples:

```bash
# Fill a form with inline content
form-filler fill form.docx "John Smith\n123 Main St\njohn@example.com" filled.docx

# Fill using content from a file
form-filler fill form.docx content.txt filled.docx --from-file

# Fill with custom field mappings
form-filler fill form.docx content.txt filled.docx --from-file --mappings mappings.json

# Fill using Anthropic Claude for advanced understanding
form-filler fill form.docx content.txt filled.docx --from-file --provider anthropic --api-key your_api_key --model claude-3-haiku-20240307
```

## Batch Processing

For batch processing multiple documents, use the `batch` subcommand:

```bash
form-filler batch [OPTIONS] [BATCH_CONFIG_FILE]
```

#### Options:

| Option           | Description                                            |
|------------------|--------------------------------------------------------|
| `--model`        | LLM model to use (default: "llama3.2:3b").             |
| `--provider`     | AI provider to use (default: "ollama").                |
| `--api-key`      | API key for the AI provider (if needed).               |
| `--api-base`     | Base URL for the AI provider API (if custom).          |
| `--output-dir`   | Directory to save output files (default: current dir). |

#### Batch Configuration Format:

```json
{
  "jobs": [
    {
      "input_file": "document1.pdf",
      "form_file": "form1.docx",
      "output_file": "output1.docx",
      "translate": true,
      "provider": "anthropic",
      "model": "claude-3-haiku-20240307",
      "api_key": "your_api_key"
    },
    {
      "input_file": "document2.png",
      "form_file": "form2.docx",
      "output_file": "output2.docx",
      "provider": "ollama",
      "model": "llama3.2:3b"
    }
  ]
}
```

You can specify AI provider settings for each job individually, or use the command-line options to set defaults for all jobs.

#### Example:

```bash
# Process a batch of documents
form-filler batch batch_config.json

# Process with OpenAI models
form-filler batch batch_config.json --provider openai --api-key your_api_key --model gpt-4 --output-dir ./processed

# Process with DeepSeek models
form-filler batch batch_config.json --provider deepseek --api-key your_api_key --model deepseek-chat-v2 --output-dir ./processed

# Process with Gemini models
form-filler batch batch_config.json --provider gemini --api-key your_api_key --model gemini-1.5-pro-latest --output-dir ./processed
```

## Environment Variables

The following environment variables can be used to configure the Form Filler CLI:

| Variable                | Description                                       |
|-------------------------|---------------------------------------------------|
| `FORM_FILLER_MODEL`     | Default LLM model to use.                         |
| `FORM_FILLER_PROVIDER`  | Default AI provider to use.                       |
| `FORM_FILLER_API_KEY`   | API key for the configured AI provider.           |
| `FORM_FILLER_API_BASE`  | Base URL for the AI provider's API.               |
| `FORM_FILLER_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR).      |
| `FORM_FILLER_OLLAMA_URL`| URL for Ollama API (default: http://localhost:11434). |

## Return Codes

| Code | Description                                                |
|------|------------------------------------------------------------|
| 0    | Success                                                    |
| 1    | General error                                              |
| 2    | Input/output file error                                    |
| 3    | Processing error                                           |
| 4    | Model/API error                                            |
| 5    | Configuration error                                        |

## Progress Bar Features

All commands in the form-filler tool now include progress indicators to provide visual feedback during long-running operations:

1. **Spinners for Indeterminate Progress**:
   - Used for operations with unknown duration
   - Shows elapsed time while processing
   - Displays colorful status updates

2. **Progress Bars for Batch Operations**:
   - Visual display of overall progress
   - Shows percentage complete
   - Displays estimated time remaining
   - Includes completed/total count

3. **Enhanced Batch Processing UI**:
   - Shows real-time job status
   - Indicates successful vs. failed operations
   - Provides per-job processing time

This feature makes the tool more user-friendly by providing immediate visual feedback and timing information, especially useful for longer operations or batch processing.
