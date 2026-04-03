import os
import ast
import re

# ==========================================
# 配置区
# ==========================================
TARGET_DIR = "./mission_sim"  # 需要扫描的代码目录
AUTO_FIX = True              # 是否开启自动更正（建议先设为 False 看扫描结果）

# 需要强制定向修复的专有名词字典 (错用的驼峰 -> 正确的全大写)
ACRONYM_FIXES = {
    r'Gnc': 'GNC',
    r'Isl': 'ISL',
    r'Lvlh': 'LVLH',
    r'Vvlh': 'VVLH',
    r'Stm': 'STM',
    r'CwDynamics': 'CWDynamics',
    r'Lqr': 'LQR',
    r'Mpc': 'MPC'
}

# ==========================================
# 阶段一：AST 语法树扫描器 (找出风格问题)
# ==========================================
class NamingConventionChecker(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.warnings = []

    def visit_ClassDef(self, node):
        # 检查大驼峰 PascalCase
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
            self.warnings.append(f"[{self.filename}:{node.lineno}] 类名 '{node.name}' 不符合 PascalCase 规范。")
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # 排除魔法方法如 __init__
        if not node.name.startswith('__'):
            # 检查蛇形 snake_case
            if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                self.warnings.append(f"[{self.filename}:{node.lineno}] 函数/方法名 '{node.name}' 不符合 snake_case 规范。")
        self.generic_visit(node)

def scan_file_ast(filepath):
    """使用 AST 解析代码并检查命名规范"""
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
            checker = NamingConventionChecker(filepath)
            checker.visit(tree)
            for warning in checker.warnings:
                print(f"⚠️ {warning}")
        except SyntaxError:
            print(f"❌ 语法错误，无法解析文件: {filepath}")

# ==========================================
# 阶段二：定向正则修复器 (修正缩写大小写)
# ==========================================
def fix_acronyms_in_file(filepath):
    """读取文件，执行定向词汇替换，并覆盖保存"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes_made = 0

    for wrong, right in ACRONYM_FIXES.items():
        # 使用正则，加上 \b 边界匹配，防止误伤（例如防止把 'ignc' 变成 'iGNC'）
        # 这里主要针对大写的错误驼峰，例如 PlatformGncMode
        pattern = re.compile(r'\b([A-Za-z0-9]*)' + wrong + r'([A-Za-z0-9]*)\b')
        
        def replace_match(match):
            return match.group(1) + right + match.group(2)
        
        new_content, count = pattern.subn(replace_match, content)
        if count > 0:
            content = new_content
            changes_made += count

    if changes_made > 0 and AUTO_FIX:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 已自动修复 {filepath} 中的 {changes_made} 处缩写大小写错误。")
    elif changes_made > 0 and not AUTO_FIX:
        print(f"🔎 发现 {filepath} 中有 {changes_made} 处缩写错误可修复（未开启 AUTO_FIX）。")

# ==========================================
# 主运行入口
# ==========================================
def main():
    print("🚀 开始 MCPC 架构代码审查...")
    for root, dirs, files in os.walk(TARGET_DIR):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                # 1. 执行风格扫描
                scan_file_ast(filepath)
                # 2. 执行缩写修复
                fix_acronyms_in_file(filepath)
                
    print("✨ 审查完成！")
    if not AUTO_FIX:
        print("💡 提示：将脚本中的 AUTO_FIX 设为 True 即可自动应用缩写更正。")

if __name__ == "__main__":
    main()
