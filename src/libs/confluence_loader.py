"""Confluence document loader — loads all spaces into LangChain documents."""
from typing import List, Any
from langchain_community.document_loaders import ConfluenceLoader
from atlassian import Confluence


def load_confluence_documents(url: str, url_extended: str, username: str, token: str) -> List[Any]:
    confluence = Confluence(url=url, username=username, password=token, cloud=True)
    spaces = confluence.get_all_spaces(start=0, limit=100)
    documents = []

    for space in spaces.get("results", []):
        space_key = space["key"]
        print(f"[Confluence] Loading space: {space_key}")
        try:
            loader = ConfluenceLoader(
                url=url_extended,
                username=username,
                api_key=token,
                space_key=space_key,
            )
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_type"] = "confluence"
                doc.metadata["source_file"] = doc.metadata.get("title", space_key)
            documents.extend(docs)
            print(f"[Confluence] Loaded {len(docs)} docs from space {space_key}")
        except Exception as e:
            print(f"[Confluence] ERROR loading space {space_key}: {e}")

    print(f"[Confluence] Total docs loaded: {len(documents)}")
    return documents
