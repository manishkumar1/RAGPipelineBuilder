"""SharePoint document loader via Microsoft Graph API."""
import io
import os
import tempfile
from typing import List, Any

import requests
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_core.documents import Document


def _get_graph_token(tenant_id: str, client_id: str, client_secret: str) -> str:
    """Obtain an app-only access token from Azure AD."""
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    resp = requests.post(url, data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default",
    })
    resp.raise_for_status()
    return resp.json()["access_token"]


def _list_drive_files(token: str, site_id: str) -> List[dict]:
    """Recursively list all files in the SharePoint site's default drive."""
    headers = {"Authorization": f"Bearer {token}"}
    files = []

    def _recurse(url: str):
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("value", []):
            if "file" in item:
                files.append(item)
            elif "folder" in item:
                child_url = (
                    f"https://graph.microsoft.com/v1.0/sites/{site_id}"
                    f"/drive/items/{item['id']}/children"
                )
                _recurse(child_url)
        if "@odata.nextLink" in data:
            _recurse(data["@odata.nextLink"])

    _recurse(f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root/children")
    return files


def _download_and_load(token: str, item: dict) -> List[Document]:
    """Download a file and load it into LangChain documents."""
    headers = {"Authorization": f"Bearer {token}"}
    download_url = item.get("@microsoft.graph.downloadUrl") or item.get("downloadUrl")
    if not download_url:
        return []

    name: str = item.get("name", "unknown")
    ext = os.path.splitext(name)[1].lower()

    supported = {".pdf", ".txt", ".docx", ".xlsx", ".csv"}
    if ext not in supported:
        return []

    resp = requests.get(download_url, headers=headers)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif ext == ".txt":
            loader = TextLoader(tmp_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif ext == ".xlsx":
            loader = UnstructuredExcelLoader(tmp_path)
        else:
            return []

        docs = loader.load()
        for doc in docs:
            doc.metadata["source_type"] = "sharepoint"
            doc.metadata["source_file"] = name
        return docs
    except Exception as e:
        print(f"[SharePoint] ERROR loading {name}: {e}")
        return []
    finally:
        os.unlink(tmp_path)


def load_sharepoint_documents(
    tenant_id: str,
    client_id: str,
    client_secret: str,
    site_id: str,
) -> List[Document]:
    print(f"[SharePoint] Connecting to site: {site_id}")
    token = _get_graph_token(tenant_id, client_id, client_secret)
    files = _list_drive_files(token, site_id)
    print(f"[SharePoint] Found {len(files)} files")

    documents = []
    for item in files:
        docs = _download_and_load(token, item)
        documents.extend(docs)
        if docs:
            print(f"[SharePoint] Loaded {len(docs)} chunks from {item.get('name')}")

    print(f"[SharePoint] Total docs loaded: {len(documents)}")
    return documents
