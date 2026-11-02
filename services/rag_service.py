import os
import glob
import chromadb

class HRPoplicyRAGService:
    def __init__(self, policies_dir="docs/hr_policies", db_path="chroma_db"):
        self.policies_dir = policies_dir
        self.db_path = db_path
        self.collection_name = "apparel_hr_policies"
        self._init_db()

    def _init_db(self):
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            # Index documents if collection is empty
            if self.collection.count() == 0:
                self._index_policy_documents()
        except Exception as e:
            print(f"Warning: ChromaDB initialization error: {e}")
            self.client = None
            self.collection = None

    def _index_policy_documents(self):
        """Indexes all markdown files in policies_dir into ChromaDB."""
        if not os.path.exists(self.policies_dir):
            return

        md_files = glob.glob(os.path.join(self.policies_dir, "*.md"))
        docs = []
        metadatas = []
        ids = []

        chunk_id = 0
        for fpath in md_files:
            fname = os.path.basename(fpath)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            # Split by markdown headers (sections)
            sections = content.split("## ")
            doc_title = sections[0].replace("# ", "").strip() if len(sections) > 0 else fname

            for sec in sections[1:]:
                lines = sec.strip().split("\n")
                sec_title = lines[0].strip()
                sec_body = "\n".join(lines[1:]).strip()
                full_chunk_text = f"Document: {doc_title}\nSection: {sec_title}\nContent:\n{sec_body}"

                chunk_id += 1
                docs.append(full_chunk_text)
                metadatas.append({
                    "source": fname,
                    "document_title": doc_title,
                    "section_title": sec_title
                })
                ids.append(f"policy_chunk_{chunk_id}")

        if docs and self.collection:
            self.collection.add(
                documents=docs,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Indexed {len(docs)} policy sections into ChromaDB collection '{self.collection_name}'.")

    def query_policy(self, query_text, n_results=3):
        """
        Retrieves top matching policy excerpts for the query.
        Returns a formatted string suitable for LLM context.
        """
        if not self.collection:
            return self._fallback_text_search(query_text)

        try:
            res = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]

            if not docs:
                return "No directly matching policy documents found."

            formatted_snippets = []
            for doc, meta in zip(docs, metas):
                formatted_snippets.append(
                    f"--- Source: {meta.get('document_title', 'Policy')} > {meta.get('section_title', 'Section')} ({meta.get('source', '')}) ---\n{doc}"
                )
            return "\n\n".join(formatted_snippets)
        except Exception as e:
            print(f"Chroma query failed, falling back to text search: {e}")
            return self._fallback_text_search(query_text)

    def _fallback_text_search(self, query_text):
        """Simple keyword matching fallback if vector engine is unavailable."""
        if not os.path.exists(self.policies_dir):
            return "Policy documentation directory not found."

        query_terms = [t.lower() for t in query_text.split() if len(t) > 3]
        snippets = []

        for fpath in glob.glob(os.path.join(self.policies_dir, "*.md")):
            with open(fpath, "r", encoding="utf-8") as f:
                txt = f.read()
            # Simple check
            matched_count = sum(1 for term in query_terms if term in txt.lower())
            if matched_count > 0:
                snippets.append(f"--- Document: {os.path.basename(fpath)} ---\n{txt[:600]}...")

        return "\n\n".join(snippets[:3]) if snippets else "No policy matching the query keywords."
