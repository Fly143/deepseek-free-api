from fastapi import HTTPException


def select_tools(tools, choice=None):
    if choice is None:
        choice = "auto"
    selected_name = None
    if isinstance(choice, dict):
        kind = choice.get("type")
        if kind in ("function", "tool"):
            function = choice.get("function") or {}
            selected_name = choice.get("name") or (function.get("name") if isinstance(function, dict) else None)
            if not isinstance(selected_name, str) or not selected_name:
                raise HTTPException(400, detail="指定的工具名称无效")
            choice = "required"
        elif kind in ("auto", "none", "any", "required"):
            choice = "required" if kind == "any" else kind
        else:
            raise HTTPException(400, detail="不支持的工具选择方式")
    if not isinstance(choice, str) or choice not in ("none", "auto", "required"):
        raise HTTPException(400, detail="不支持的工具选择方式")
    if choice == "none":
        return None, False
    if not isinstance(tools, list):
        tools = []
    if selected_name:
        tools = [tool for tool in tools if isinstance(tool, dict) and isinstance(tool.get("function"), dict) and tool["function"].get("name") == selected_name]
        if not tools:
            raise HTTPException(400, detail="指定的工具不在工具列表中")
    if choice == "required" and not tools:
        raise HTTPException(400, detail="要求调用工具时必须提供工具列表")
    return tools or None, choice == "required"
