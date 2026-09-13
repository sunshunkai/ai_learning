---
name: python-project-starter
description: Scaffold a small Python project with a documented entry point and minimal dependencies.
triggers: python project, python cli, scaffold, 新建python项目, python项目
allowed_roles: developer
version: 1.0.0
---

# Python Project Starter

Use this skill when the user asks to create or scaffold a small Python project.

## Steps

1. Identify the project name and entry-point behavior from the request.
2. Create or propose a small layout with `main.py` and `requirements.txt`.
3. Include a module docstring, a `main()` function, and the
   `if __name__ == "__main__":` guard.
4. Keep dependencies minimal and explain every dependency that is added.
5. Verify generated file contents before summarizing the result.

## Output

Return a concise summary with:

- created or proposed files
- the command used to run the project
- any assumptions that still need confirmation
