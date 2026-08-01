# GenericLoader + FileSystemBlobLoader + PyMuPDF4LLMParser (In-Depth Summary)

This approach follows LangChain's **modular document loading architecture**, where **loading files** and **parsing files** are separated into independent components.

Instead of having one loader that does everything, the work is divided into three parts:

```text
PDF File
   │
   ▼
FileSystemBlobLoader
   │
   ▼
Blob (Raw PDF)
   │
   ▼
PyMuPDF4LLMParser
   │
   ▼
LangChain Documents
```

---

# Why LangChain uses this architecture

Imagine tomorrow your PDF is stored in

* Local folder
* Amazon S3
* Google Drive
* Dropbox
* Azure Blob Storage

The **parsing logic remains exactly the same**.

Only the **Blob Loader** changes.

Similarly, if tomorrow you want to parse

* PDF
* Word
* HTML
* Markdown

the **Blob Loader** stays the same.

Only the **Parser** changes.

This separation makes the architecture reusable and extensible.

---

# Component 1 : FileSystemBlobLoader

```python
blob_loader = FileSystemBlobLoader(
    path="...",
    glob="*.pdf"
)
```

## Purpose

Its only responsibility is

> **Find files and read them as raw binary data (Blobs).**

It does **not**

* understand PDFs
* extract text
* detect images
* detect tables

It simply reads files.

Example

```text
Folder

├── paper1.pdf
├── paper2.pdf
├── paper3.pdf
```

↓

Produces

```text
Blob1

Blob2

Blob3
```

Each Blob contains

* binary bytes
* file path
* MIME type
* metadata

Nothing more.

---

# Component 2 : GenericLoader

```python
loader = GenericLoader(
    blob_loader=...,
    blob_parser=...
)
```

GenericLoader is the coordinator.

It simply says

```text
Load Blob

↓

Send Blob to Parser

↓

Collect Documents

↓

Return List[Document]
```

It knows nothing about PDFs.

Its job is orchestration.

Internally

```python
blobs = blob_loader.yield_blobs()

documents = []

for blob in blobs:

    docs = parser.parse(blob)

    documents.extend(docs)

return documents
```

---

# Component 3 : PyMuPDF4LLMParser

This is the actual intelligence.

```python
PyMuPDF4LLMParser()
```

This parser converts

```text
Raw PDF

↓

Markdown

↓

LangChain Documents
```

---

# Internal Working of Parser

For every page

```text
Open PDF

↓

Read Metadata

↓

Analyze Layout

↓

Extract Text

↓

Detect Tables

↓

Detect Images

↓

Run OCR (if needed)

↓

Convert Everything to Markdown

↓

Create Document
```

Finally

```python
Document(
    page_content=...,
    metadata=...
)
```

---

# How Normal Text is Handled

Original PDF

```text
1 Introduction

Transformer is...
```

Parser produces

```markdown
# 1 Introduction

Transformer is...
```

Notice

* Heading preserved
* Paragraph preserved
* Reading order preserved

---

# How Headings are Handled

Large font

↓

Detected as heading

↓

Markdown

```markdown
# Attention Is All You Need
```

Instead of

```text
Attention Is All You Need
```

LLMs understand Markdown headings much better.

---

# How Lists are Handled

Original PDF

```text
• Encoder

• Decoder

• Attention
```

Parser

↓

Markdown

```markdown
- Encoder

- Decoder

- Attention
```

---

# How Tables are Handled

Suppose PDF contains

```text
-----------------------------------
Model        BLEU
-----------------------------------
Transformer   28.4
GNMT          24.6
-----------------------------------
```

Parser first detects

```text
This region is a table
```

Instead of OCR

↓

Layout Analysis

↓

Detect Rows

↓

Detect Columns

↓

Generate Markdown

Output

```markdown
| Model | BLEU |
|-------|------|
| Transformer | 28.4 |
| GNMT | 24.6 |
```

### Why this is important

Normal OCR gives

```text
Model BLEU Transformer 28.4 GNMT 24.6
```

The relationships between cells are lost.

Markdown preserves

* rows
* columns
* headers
* structure

This makes table-based question answering much more accurate.

---

# How Images are Handled

Suppose page contains

```text
Transformer Architecture Figure
```

Parser detects

```text
Figure
```

↓

Runs OCR

↓

Extracts visible text

Example

Original image

```text
Output Probabilities

Add & Norm

Feed Forward

Multi-Head Attention
```

Output

```markdown
<!-- Start of picture -->

Output Probabilities

Add & Norm

Feed Forward

Multi-Head Attention

<!-- End of picture -->
```

Notice

The parser does **not**

* save PNG
* save JPEG
* preserve arrows
* preserve colors
* preserve boxes

Only readable text is extracted.

---

# If Image Has No Text

Suppose image is

```text
Company Logo
```

OCR finds

```text
Nothing
```

Then

Nothing meaningful is added to page_content.

The parser does **not** generate captions like

> "This is a company logo."

Image understanding is outside the scope of this parser.

---

# OCR Handling

Pages with searchable text

```text
PDF Text Layer

↓

Direct Extraction
```

Pages containing scanned text

```text
Image

↓

Tesseract OCR

↓

Markdown Text
```

This is why you saw

```
Using Tesseract for OCR processing
```

Only pages requiring OCR invoke Tesseract.

---

# Metadata Extraction

Every Document also contains metadata.

Example

```python
{
    "page": 5,
    "total_pages": 15,
    "source": "...attention_paper.pdf",
    "creator": "LaTeX",
    "producer": "pdfTeX"
}
```

Metadata helps later during retrieval by identifying the source page and document.

---

# Final Output

After processing a 15-page PDF

```python
docs
```

contains

```text
[
 Document(Page1),
 Document(Page2),
 Document(Page3),
 ...
 Document(Page15)
]
```

Each `Document` consists of

```python
Document(
    page_content=Markdown representation of the page,
    metadata=Information about the page
)
```

---

# Advantages of This Approach

* **Modular Design:** Loading and parsing are separate, making it easy to swap file sources or parsers.
* **LLM-Friendly Output:** Converts PDFs into structured Markdown instead of plain text.
* **Rich Structure:** Preserves headings, lists, tables, and reading order.
* **Table Preservation:** Converts tables into Markdown rather than flattening them.
* **OCR Support:** Extracts text from scanned pages and figures using Tesseract.
* **Metadata:** Keeps page numbers, source paths, and document information for retrieval.
* **Scalable:** The same parser can work with local files, cloud storage, or any other blob source.

---

# Complete Workflow

```text
                    PDF File
                        │
                        ▼
              FileSystemBlobLoader
                        │
                        ▼
                Blob (Raw PDF Bytes)
                        │
                        ▼
                  GenericLoader
                        │
                        ▼
               PyMuPDF4LLMParser
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
   Extract Text     Detect Tables    Detect Images
        │               │                │
        ▼               ▼                ▼
    Markdown      Markdown Tables     OCR Text
        └───────────────┼────────────────┘
                        ▼
              LangChain Document
                        │
                        ▼
               List[Document]
                        │
                        ▼
              Text Splitter (optional)
                        │
                        ▼
                  Embeddings
                        │
                        ▼
                 Vector Database
                        │
                        ▼
                 Retriever + LLM
```

### In one sentence

**`GenericLoader + FileSystemBlobLoader + PyMuPDF4LLMParser` is a modular LangChain pipeline where the Blob Loader reads raw PDF files, the Parser intelligently converts each page into LLM-friendly Markdown (preserving text, tables, and OCR-extracted figure text), and the output is a list of LangChain `Document` objects ready for chunking, embedding, and retrieval in RAG systems.**
