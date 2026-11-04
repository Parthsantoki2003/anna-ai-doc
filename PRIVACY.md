# Privacy Policy for DocuMind

**Effective Date:** November 4, 2026  
**Last Updated:** November 4, 2026

DocuMind ("we", "our", or "the App") is an intelligent document assistance application built for Anna OS. We respect your privacy and are committed to protecting the information you process through our service. This Privacy Policy explains how data is handled when you use DocuMind.

---

## 1. Information We Process

When you use DocuMind, the following data may be processed:
- **Uploaded Documents (PDFs):** Files you explicitly upload to the App to enable grounded Q&A, search, and synthesis.
- **Questions & Queries:** The natural language prompts and questions you submit regarding your uploaded documents.
- **Anonymous Session Identifiers:** A randomly generated client session identifier (`user_id`) stored locally on your device to isolate your workspace and prevent document mixing across sessions.

---

## 2. How Your Information Is Used

Your data is used strictly to provide the core functionality of the App:
- **Text Extraction & Embedding:** Uploaded documents are parsed and converted into mathematical vector representations (embeddings) for similarity matching.
- **Grounded Retrieval & Synthesis:** Relevant text excerpts are retrieved to answer your questions accurately using language model inference (Groq API or native Anna OS platform LLM).
- **Session Isolation:** All vector embeddings and metadata are strictly filtered by your session identifier to ensure complete multi-tenant privacy.

---

## 3. Data Storage & Retention

- **Ephemeral Storage:** Document vectors and metadata are stored in isolated vector collections only as long as you maintain them in your workspace.
- **User-Controlled Deletion:** When you click the delete icon for any document in the DocuMind interface, all associated text chunks, embeddings, and metadata records are **immediately and permanently deleted** from the vector index.
- **No Model Training:** Your uploaded documents, queries, and generated answers are **never used to train, retrain, or fine-tune** any public or private artificial intelligence models.

---

## 4. Third-Party Services & Infrastructure

DocuMind uses reputable infrastructure providers to operate:
- **Google Cloud Platform (Cloud Run):** Hosts the secure backend API in containerized, isolated environments.
- **Qdrant Cloud:** Provides managed, encrypted vector search infrastructure.
- **Groq & Anna OS LLM:** Performs inference solely to synthesize answers from your provided document context. Requests sent to inference providers are processed in real-time and not retained for training.

---

## 5. Security

We implement standard security practices, including:
- Encryption in transit via HTTPS / TLS for all API communication.
- Cryptographic isolation of multi-tenant document catalogs by unique session keys.
- Secret management practices ensuring API keys and credentials are never exposed to the client.

---

## 6. Your Rights & Choices

You have full control over your data:
- You can delete any uploaded document at any time using the in-app delete feature.
- You can clear your local session data and start fresh at any time by resetting your browser storage or clicking "New Conversation".

---

## 7. Contact & Support

If you have questions, feedback, or concerns regarding this Privacy Policy or data handling in DocuMind, please open an issue on our GitHub repository:  
[https://github.com/Parthsantoki2003/anna-ai-doc/issues](https://github.com/Parthsantoki2003/anna-ai-doc/issues)
