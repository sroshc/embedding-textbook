# Embedding Textbooks with text-embedding-3-small

This guide outlines how to index textbooks using the text-embedding-3-small model.

### Textbook File Organization

- Place textbook files within folders named after the corresponding chapter (e.g., "textbookfiles/13").
- Name each file with the chapter and section number (e.g., "13.1.txt").

### Textbook File Formatting\*\*

- Separate paragraphs with the word "ENDLINE" on a new line.
- If a paragraph is too long, divide it into smaller, more manageable paragraphs.
- Do not use "ENDLINE" after the final paragraph of a chapter section.

**Example:**

Once T. H. Morgan’s group showed that genes exist as parts of chromosomes. ENDLINE
In 1928, a British medical officer named Frederick Griffith was trying to experiment. ENDLINE
The brief but celebrated partnership that solved the puzzle of DNA structure began.

### Indexing and Retrieval in Pinecone

The program pinecone_indexer.py will take input in a chapter folder, and index all the files that are made according to above guidelines. This will be inputed into
a pinecone index. For each section, they will be named in the pinecone inedx as so, the last digit representing the paragraph number of the section:

**Example:**
13.1.1
13.1.2
13.1.3

The program pinecone_serach.py will take in user input of a query, then do a similarity search with all the previous inputed indexes.

**Example:**
Enter your search query: What are the nitrogenous bases of the double helix? what are the combinations?
Most similar embedding ID: 13.1.13
Similarity score: 0.713876247

That corresponds with the 13th paragraph of section 13.1
