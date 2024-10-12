module.exports = {
    // 使用官方的 conventional 配置作为基础
    extends: ["@commitlint/config-conventional"],
    rules: {
      // 限制提交信息头部的最大长度为150个字符
      "header-max-length": [2, "always", 150],
      "type-enum": [
        2,
        "always",
        ["build", "chore", "ci", "docs", "feat", "fix", "perf", "refactor", "revert", "style", "test", "Release-As"]
      ]
    }
  };
