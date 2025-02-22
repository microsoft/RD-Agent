# CI 检查

`.github/workflows/ci.yml`配置了提交时自动运行`Makefile`: 91~103行的命令，可以在这调整执行的命令

在`.env`中设置`USE_CHAT_CACHE=True`可以让第二次修复快一些

# Rules

`pyproject.toml`中配置全局屏蔽的规则
- ruff: `[tool.ruff.lint].ignore`
- mypy: `[tool.mypy]`

## ruff rules
ruff rules 比较好修改, 大多可以自动修复

对于一些规则可以在代码中添加注释来局部屏蔽, 例如添加 `# noqa E234,ANN001`
遇到的不好修改的规则:
- 捕获异常时应该处理每一种异常，不应该统一当作`Exception`处理
- `subprogress()` 调用命令应该先判断命令是否安全
- ...

规则列表: [ruff rules](https://docs.astral.sh/ruff/rules/)

## mypy rules

Mypy检查Python中类型标注, 常遇到需要修改结构/同时修改其他文件的情况, 自动修复效果不好

局部屏蔽: `# type: ignore`

规则列表: [mypy rules](https://mypy.readthedocs.io/en/stable/error_code_list.html)

# Optimization (Maybe)

- 添加指定文件夹检查的功能
- 增加一个修改选项: 调用`vim`, 用户直接修改此部分代码
- 显示时把`Original Code`部分去掉, 直接在输出的表示修改的diff部分用`^^^^^^`在代码行下标注出错误位置，这样能更直观地观察错误修复情况
- 当前为线性执行完所有修复后交给用户检查, 可修改成 后台多线程 / 进程处理修复的任务, 终端实时展示处理完的修复让用户检查
- ...
