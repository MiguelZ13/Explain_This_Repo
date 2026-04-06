import tempfile
import os
import subprocess
import ast
import re
from dataclassses import dataclass, field
from typing import Optional

@dataclass
class ParsedChunk:
    content: str
    file_path: str
    language: str
    chunk_type: str         # "function", "class", "code_block"
    name: Optional[str] = None
    docstring: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    extra: dict = field(default_factory=dict)

class RepoParser:
    def __init__(self):
        pass

    def parse_repo(self, repo_url: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_name = os.path.basename(repo_url).replace('.git', '')
            repo_path = os.path.join(tmpdir, repo_name)
            subprocess.run(["git", "clone", repo_url, repo_path], check=True)
            
            file_list = []
            for root, _, files in os.walk(repo_path):
                for file in files:
                    file_list.append(os.path.relpath(os.path.join(root, file), repo_path))
            return {"repo_name": repo_name, "files": file_list}
        