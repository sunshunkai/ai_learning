# -*- coding: utf-8 -*-
"""
模型层结构化输出。

AgentScope 会把 Pydantic schema 转成强制工具调用，再校验和修复 JSON。
这里演示直接调用 ChatModel.generate_structured_output。

运行:
    python agentscope-demo/c02_03_structured_output.py
"""

import asyncio
import json

from pydantic import BaseModel, Field

from common import get_model, make_user


class TravelPlan(BaseModel):
    """一日杭州旅行计划。"""

    title: str = Field(description="计划标题")
    city: str = Field(description="目的地城市")
    total_hours: float = Field(description="总时长，单位小时")
    activities: list[str] = Field(description="按时间顺序排列的 3-5 个活动")
    suitable_for_family: bool = Field(description="是否适合家庭出游")


class CodeReview(BaseModel):
    """代码审查结构化结果。"""

    overall_rating: int = Field(ge=1, le=5, description="1-5 分")
    must_fix: list[str] = Field(description="必须修复的问题")
    suggestions: list[str] = Field(description="改进建议")


async def main() -> None:
    print("=== 1. 提取旅行计划 ===")
    model = get_model(stream=False, thinking_enable=False, max_tokens=1500)
    response = await model.generate_structured_output(
        messages=[
            make_user(
                "请规划一个杭州亲子一日游：上午去西湖，中午在知味观吃饭，"
                "下午去杭州动物园，晚上逛河坊街。总时长约 9 小时。"
            )
        ],
        structured_model=TravelPlan,
    )
    print(json.dumps(response.content, ensure_ascii=False, indent=2))

    print("\n=== 2. 提取代码审查结果 ===")
    response = await model.generate_structured_output(
        messages=[
            make_user(
                "我写了一个登录接口，但没有参数校验，也没有做数据库索引。"
                "另外密码是明文保存的。请给这次代码实现打一个结构化审查结论。"
            )
        ],
        structured_model=CodeReview,
    )
    review = CodeReview.model_validate(response.content)
    print(f"评分: {review.overall_rating}/5")
    print("必须修复:")
    for item in review.must_fix:
        print(f"  - {item}")
    print("改进建议:")
    for item in review.suggestions:
        print(f"  - {item}")


if __name__ == "__main__":
    asyncio.run(main())
