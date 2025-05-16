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
| `--openai`       | Use OpenAI models instead of local Ollama.             |
| `--translate`    | Translate the input document if it's in Vietnamese.    |

#### Examples:

```bash
# Process a document using default Ollama model
form-filler process scan.pdf form.docx output.docx

# Process with translation
form-filler process vietnamese_doc.png form.docx output.docx --translate

# Process using OpenAI models
form-filler process scan.pdf form.docx output.docx --openai
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
| `--openai`       | Use OpenAI models instead of local Ollama.             |
| `--method`       | Extraction method: "traditional", "ai", or "auto" (default).  |
| `--output`       | Output file to save the extracted text (optional).     |

#### Examples:

```bash
# Extract text from a PDF
form-filler extract document.pdf

# Extract with specific method and save to file
form-filler extract receipt.png --method ai --output extracted.txt

# Extract using OpenAI
form-filler extract document.pdf --openai
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
| `--openai`       | Use OpenAI models instead of local Ollama.             |
| `--output-dir`   | Directory to save output files (default: current dir). |

#### Batch Configuration Format:

```json
{
  "jobs": [
    {
      "input_file": "document1.pdf",
      "form_file": "form1.docx",
      "output_file": "output1.docx",
      "translate": true
    },
    {
      "input_file": "document2.png",
      "form_file": "form2.docx",
      "output_file": "output2.docx"
    }
  ]
}
```

#### Example:

```bash
# Process a batch of documents
form-filler batch batch_config.json

# Process with OpenAI models and specific output directory
form-filler batch batch_config.json --openai --output-dir ./processed
```

## Environment Variables

The following environment variables can be used to configure the Form Filler CLI:

| Variable                | Description                                       |
|-------------------------|---------------------------------------------------|
| `FORM_FILLER_MODEL`     | Default LLM model to use.                         |
| `FORM_FILLER_OPENAI_KEY`| OpenAI API key for OpenAI model usage.            |
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
