# app/parsers.py
from typing import List

from langchain_core.output_parsers import BaseOutputParser


class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split('\n')
        return list(filter(None, lines))  # Remove empty lines
