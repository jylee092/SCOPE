"""Benign FP probe -- run ONLY the deterministic grouping stage on the benign
scenario and report whether anchors fire / how many behavior groups form.
No LLM/API. Answers the first question: do benign logs produce groups at all?"""
import sys
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from pipeline.data_loader import load_and_normalize
from pipeline.rule_matcher import load_rules, run_grouping, merge_same_anchor, merge_shared_supporting

DS = config.DATASET_FOLDER / "normal" / "Normal.json"
config.configure_dataset(DS)

df = load_and_normalize(str(config.DATASET_FILE))
print(f"events loaded: {len(df)}")

rule_list = load_rules(config.RULE_FOLDER)
print(f"rules loaded : {len(rule_list)}")

groups = run_grouping(
    df=df, rule_list=rule_list,
    before_sec=config.GROUPING_BEFORE_SEC, after_sec=config.GROUPING_AFTER_SEC,
    hop_up=config.GROUPING_HOP_UP, hop_down=config.GROUPING_HOP_DOWN,
    apply_filters=config.GROUPING_APPLY_FILTER,
    use_shared_entity=config.GROUPING_USE_SHARED_ENTITY,
    max_anchors_per_rule=config.GROUPING_MAX_ANCHORS_PER_RULE,
)
print(f"groups (raw anchors)          : {len(groups)}")
groups = merge_same_anchor(groups)
print(f"groups (after same-anchor merge): {len(groups)}")
groups = merge_shared_supporting(groups, df, overlap_threshold=config.MERGE_OVERLAP_THRESHOLD)
print(f"groups (after shared merge)     : {len(groups)}")
if getattr(config, "DROP_FILTER_FAILED_GROUPS", False):
    groups = [g for g in groups if g.get("filter_passed", True)]
    print(f"groups (after filter drop)      : {len(groups)}")
cap = config.MAX_GROUPS_PER_SCENARIO
capped = min(len(groups), cap)
print(f"MAX_GROUPS_PER_SCENARIO cap     : {cap}  -> final would be {capped}")

print("\n=== groups by anchor technique_id (top 25) ===")
by_tid = Counter(g.get("technique_id", "?") for g in groups)
for tid, n in by_tid.most_common(25):
    print(f"  {n:4d}  {tid}")

print("\n=== groups by anchor rule name (top 25) ===")
by_rule = Counter((g.get("rule_name") or g.get("anchor_rule") or "?") for g in groups)
for r, n in by_rule.most_common(25):
    print(f"  {n:4d}  {str(r)[:70]}")
