"""
人机确认层（Human-in-the-loop）

功能：
1. 从 Spec YAML 中提取关键对齐指标
2. 生成供用户检查的确认清单
3. 支持交互式确认（命令行）和文件式确认（保存状态）

输出：
- confirm/<factor_name>/confirmation.yaml
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml


CONFIRM_DIR = Path(__file__).parent.parent / "confirm"


def validate_and_confirm(
    factor_name: str,
    spec_yaml: dict,
    auto_confirm: bool = False,
    interactive: bool = True,
) -> bool:
    """
    人机确认主入口

    Args:
        factor_name: 因子名称
        spec_yaml: 已加载的 Spec YAML dict
        auto_confirm: 是否自动跳过确认（用于批量跑历史已确认因子）
        interactive: 是否在命令行交互询问

    Returns:
        bool: True = 确认通过，False = 被拒绝或需要修改
    """
    confirm_dir = CONFIRM_DIR / factor_name
    confirm_dir.mkdir(parents=True, exist_ok=True)
    confirm_path = confirm_dir / "confirmation.yaml"

    # 1. 提取关键对齐指标
    checklist = _extract_checklist(spec_yaml)

    # 2. 如果 auto_confirm=True，自动通过（用于全自动流水线）
    if auto_confirm:
        _save_confirmation(factor_name, spec_yaml, "confirmed", "auto-confirmed")
        print(f"✅ {factor_name} 已自动确认")
        return True

    # 3. 输出确认清单
    print(f"\n{'='*60}")
    print(f"🤖 Agent: {factor_name} 的 Spec 已生成，请确认以下关键对齐指标：")
    print(f"{'='*60}\n")

    for i, item in enumerate(checklist, 1):
        status = "✅" if item.get("ok") else "⚠️"
        print(f"{i}. {status} {item['desc']}")
        if "detail" in item:
            print(f"   详情: {item['detail']}")

    # 4. 风险提示
    warnings = spec_yaml.get("risk_warnings", [])
    if warnings:
        print(f"\n🔴 风险提示:")
        for w in warnings:
            emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(w["level"], "⚪")
            print(f"   {emoji} [{w['level']}] {w['item']}: {w['description']}")

    # 5. 交互确认
    if interactive and not auto_confirm:
        print(f"\n{'-'*60}")
        print("请选择操作：")
        print("  [y] 确认通过，进入 YOLO 执行")
        print("  [n] 拒绝，退出")
        print("  [e] 需要修改，提示具体修改意见")

        try:
            choice = input("你的选择 [y/n/e]: ").strip().lower()
        except EOFError:
            # 非交互环境（如脚本运行）默认拒绝
            print("非交互环境，默认保存为 pending，请手动编辑 confirmation.yaml")
            choice = "n"

        if choice == "y":
            _save_confirmation(factor_name, spec_yaml, "confirmed", "")
            print(f"✅ {factor_name} 已确认")
            return True
        elif choice == "e":
            try:
                feedback = input("请输入修改意见: ").strip()
            except EOFError:
                feedback = ""
            _save_confirmation(factor_name, spec_yaml, "pending", feedback)
            print(f"📝 {factor_name} 已标记为待修改，请调整 Spec 后重新运行")
            return False
        else:
            _save_confirmation(factor_name, spec_yaml, "rejected", "")
            print(f"❌ {factor_name} 已拒绝")
            return False

    # 非交互模式且未 auto_confirm：保存为 pending
    _save_confirmation(factor_name, spec_yaml, "pending", "")
    print(f"⏳ {factor_name} 已保存为 pending 状态，请手动确认: {confirm_path}")
    return False


def _extract_checklist(spec_yaml: dict) -> List[Dict]:
    """从 Spec YAML 中提取关键确认项"""
    checklist = []

    factor = spec_yaml.get("factor", {})
    checklist.append({
        "desc": f"因子名称: {factor.get('name', 'N/A')} ({factor.get('name_cn', 'N/A')})",
        "ok": True,
    })
    checklist.append({
        "desc": f"因子方向: {'正向' if factor.get('direction') == 1 else '反向' if factor.get('direction') == -1 else '中性'}",
        "ok": True,
    })

    data = spec_yaml.get("data_alignment", {})
    checklist.append({
        "desc": f"PIT 模式: {'开启' if data.get('pit_mode') else '未开启'}",
        "ok": data.get("pit_mode", False),
        "detail": f"API: {data.get('api', 'N/A')}, 公告日字段: {data.get('announcement_field', 'N/A')}",
    })

    fields = data.get("fields", [])
    for f in fields:
        notes = f.get("notes", "")
        is_cum = "累计" in notes or "cumulative" in notes.lower()
        checklist.append({
            "desc": f"字段 {f['variable']}: {f['rq_field']} ({f['table']})",
            "ok": True,
            "detail": f"类型: {f.get('data_type', 'N/A')}, 备注: {notes}",
        })

    steps = spec_yaml.get("calculation_steps", [])
    for step in steps:
        action = step.get("action", "")
        if action == "filter":
            checklist.append({
                "desc": f"过滤条件: {step.get('condition', 'N/A')}",
                "ok": True,
                "detail": "请确认过滤逻辑是否与研报一致",
            })
        elif action == "transform" and step.get("method") == "diff_quarterly":
            checklist.append({
                "desc": "单季度差分: Q1保留原值，其他季度差分",
                "ok": True,
                "detail": "这是利润表累计值转单季度的标准做法",
            })
        elif action == "transform" and step.get("method") == "yoy":
            checklist.append({
                "desc": f"同比计算: lag={step.get('lag_periods', 4)} 期",
                "ok": step.get("lag_periods", 4) == 4,
                "detail": "同比应为 t vs t-4（4个季度）",
            })

    universe = spec_yaml.get("universe", {})
    checklist.append({
        "desc": f"股票池: {universe.get('primary_index', 'N/A')}",
        "ok": True,
        "detail": f"过滤: {[f['type'] for f in universe.get('filter', [])]}",
    })

    backtest = spec_yaml.get("backtest", {})
    checklist.append({
        "desc": f"回测参数: {backtest.get('rebalance', {}).get('frequency', 'N/A')}调仓, 持仓{backtest.get('holding', {}).get('count', 'N/A')}只",
        "ok": True,
    })

    return checklist


def _save_confirmation(
    factor_name: str,
    spec_yaml: dict,
    status: str,
    feedback: str,
) -> None:
    """保存确认状态到文件"""
    confirm_dir = CONFIRM_DIR / factor_name
    confirm_dir.mkdir(parents=True, exist_ok=True)
    confirm_path = confirm_dir / "confirmation.yaml"

    data = {
        "factor_name": factor_name,
        "status": status,  # confirmed / pending / rejected
        "feedback": feedback,
        "spec_version": spec_yaml.get("version_control", {}).get("spec_version", "1.0"),
        "key_checks": [item["desc"] for item in _extract_checklist(spec_yaml)],
    }

    with open(confirm_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def is_confirmed(factor_name: str) -> bool:
    """检查某个因子是否已经过人工确认"""
    confirm_path = CONFIRM_DIR / factor_name / "confirmation.yaml"
    if not confirm_path.exists():
        return False
    data = yaml.safe_load(confirm_path.read_text(encoding="utf-8"))
    return data.get("status") == "confirmed"
