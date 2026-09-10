# -*- coding: utf-8 -*-
"""
结构化输出：用 Pydantic 约束返回 JSON，并自动转成 Python 对象

运行: python langchain-demo/05_structured_output.py
"""

import json

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from common import build_model


class Movie(BaseModel):
    """一部电影的结构化信息。"""

    title: str = Field(description="电影名")
    director: str = Field(description="导演")
    year: int = Field(description="上映年份")
    genres: list[str] = Field(description="类型列表")


def main():
    model = build_model(temperature=0)
    schema = json.dumps(Movie.model_json_schema(), ensure_ascii=False)

    print("=== JsonOutputParser + Pydantic 校验 ===")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "只输出符合给定 JSON Schema 的 JSON：{schema}"),
            ("human", "介绍电影《星际穿越》。"),
        ]
    )
    chain = prompt | model | JsonOutputParser()
    raw = chain.invoke({"schema": schema})
    movie = Movie.model_validate(raw)
    print(movie.model_dump(), "\n")

    print("=== with_structured_output：让模型按 schema 输出 ===")
    structured_model = model.with_structured_output(Movie, method="json_mode")
    movie2 = structured_model.invoke(
        f"JSON Schema: {schema}\n请根据该 Schema 输出电影《盗梦空间》的信息。"
    )
    print(movie2.model_dump())


if __name__ == "__main__":
    main()
