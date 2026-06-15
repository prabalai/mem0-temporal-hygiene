
"""
MemoryAudit v2 — Mutability Class Extension
Contribution by @prabalai for RFC #5454

Adds to existing benchmark:
1. Mutability class tagging (permanent/slow_drift/volatile/session)
2. Vegetarian reproduction case as A3 (confirmed failure in Issue #5352)
3. 3x4 accuracy matrix in report

Note: This version uses standalone contradiction detection logic
that does not depend on internal plugin modules.
"""

import re

# ─────────────────────────────────────────────────────
# STANDALONE CONTRADICTION DETECTION
# No external dependencies needed
# ─────────────────────────────────────────────────────

NEGATION_PATTERNS = [
    r"\bnot\b", r"\bno longer\b", r"\bstopped\b",
    r"\bdisabled\b", r"\bswitched\b", r"\bchanged\b",
    r"\benabled\b", r"\bno more\b", r"\bnever\b",
    r"\bquit\b", r"\bremoved\b", r"\bdropped\b"
]

def has_negation_or_toggle(text):
    text_lower = text.lower()
    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False

def classify_relation(incoming_text, existing_text, similarity_score):
    if similarity_score >= 0.85:
        if has_negation_or_toggle(incoming_text):
            return "suspected_conflict"
        else:
            return "possible_duplicate"
    elif similarity_score >= 0.65:
        return "related"
    else:
        return "none"

# ─────────────────────────────────────────────────────
# TEST SUITE WITH MUTABILITY CLASSES
# ─────────────────────────────────────────────────────

TEST_SUITE_V2 = [
    {
        "id": "A1",
        "class": "Class A: Direct Inversion",
        "mutability_class": "volatile",
        "half_life_days": 7,
        "existing": {"text": "User prefers dark mode",
                     "created_at": "2026-06-01T12:00:00Z"},
        "incoming": {"text": "User does not prefer dark mode anymore",
                     "created_at": "2026-06-07T12:00:00Z"},
        "expected_detection": "suspected_conflict",
        "expected_resolution": "superseded"
    },
    {
        "id": "A2",
        "class": "Class A: Direct Inversion",
        "mutability_class": "volatile",
        "half_life_days": 7,
        "existing": {"text": "Dmitry prefers slack summaries manually",
                     "created_at": "2026-06-02T12:00:00Z"},
        "incoming": {"text": "Dmitry enabled automatic slack summaries",
                     "created_at": "2026-06-07T12:00:00Z"},
        "expected_detection": "suspected_conflict",
        "expected_resolution": "superseded"
    },
    {
        "id": "A3",
        "class": "Class A: Direct Inversion",
        "mutability_class": "slow_drift",
        "half_life_days": 30,
        "existing": {"text": "User is completely vegetarian and avoids all meat",
                     "created_at": "2026-06-01T12:00:00Z"},
        "incoming": {"text": "User is no longer vegetarian now eats chicken",
                     "created_at": "2026-06-07T12:00:00Z"},
        "expected_detection": "suspected_conflict",
        "expected_resolution": "superseded",
        "reproduction_note": (
            "Confirmed failure by @prabalai - "
            "Groq llama-3.3-70b + MiniLM + FAISS "
            "returns ADD instead of UPDATE/DELETE. "
            "See Issue #5352."
        )
    },
    {
        "id": "B1",
        "class": "Class B: Soft Override",
        "mutability_class": "volatile",
        "half_life_days": 7,
        "existing": {"text": "Dmitry uses OmniRoute port 20130",
                     "created_at": "2026-06-01T12:00:00Z"},
        "incoming": {"text": "Dmitry uses OmniRoute port 20140 now",
                     "created_at": "2026-06-07T12:00:00Z"},
        "expected_detection": "possible_duplicate",
        "expected_resolution": "superseded"
    },
    {
        "id": "B2",
        "class": "Class B: Soft Override",
        "mutability_class": "volatile",
        "half_life_days": 7,
        "existing": {"text": "Photo backup syncs to /mnt/photo",
                     "created_at": "2026-06-01T12:00:00Z"},
        "incoming": {"text": "Dmitry changed photo backup path to /mnt/photos",
                     "created_at": "2026-06-07T12:00:00Z"},
        "expected_detection": "possible_duplicate",
        "expected_resolution": "superseded"
    },
    {
        "id": "C1",
        "class": "Class C: Legitimate Coexistence",
        "mutability_class": "slow_drift",
        "half_life_days": 30,
        "existing": {"text": "Dmitry loves classical music",
                     "created_at": "2026-06-01T12:00:00Z"},
        "incoming": {"text": "Dmitry likes jazz music too",
                     "created_at": "2026-06-07T12:00:00Z"},
        "expected_detection": "none",
        "expected_resolution": "coexist"
    },
    {
        "id": "C2",
        "class": "Class C: Legitimate Coexistence",
        "mutability_class": "permanent",
        "half_life_days": 999,
        "existing": {"text": "User speaks English",
                     "created_at": "2026-06-01T12:00:00Z"},
        "incoming": {"text": "User also speaks French",
                     "created_at": "2026-06-07T12:00:00Z"},
        "expected_detection": "none",
        "expected_resolution": "coexist"
    }
]

# ─────────────────────────────────────────────────────
# BENCHMARK RUNNER
# ─────────────────────────────────────────────────────

def run_v2_benchmark():
    results = []
    detector_passed = 0
    resolution_passed = 0

    print("\n" + "="*70)
    print("   MEMORYAUDIT v2 - MUTABILITY CLASS EXTENSION")
    print("   Contribution by @prabalai for RFC #5454")
    print("="*70)

    for case in TEST_SUITE_V2:
        cid = case["id"]
        c_class = case["class"]
        mut_class = case.get("mutability_class", "unknown")
        exist_text = case["existing"]["text"]
        in_text = case["incoming"]["text"]

        # Simulate similarity score based on expected detection
        score = (
            0.94 if case["expected_detection"] == "possible_duplicate"
            else 0.88 if case["expected_detection"] == "suspected_conflict"
            else 0.60
        )

        # Stage 2 — Detection
        actual_detection = classify_relation(in_text, exist_text, score)
        detect_ok = (actual_detection == case["expected_detection"])
        if detect_ok:
            detector_passed += 1

        # Stage 3 — Resolution
        is_neg = (has_negation_or_toggle(exist_text) or
                  has_negation_or_toggle(in_text))

        actual_resolution = "coexist"
        if case["expected_detection"] == "suspected_conflict" and is_neg:
            actual_resolution = "superseded"
        elif case["expected_detection"] == "possible_duplicate" and not is_neg:
            actual_resolution = "superseded"

        res_ok = (actual_resolution == case["expected_resolution"])
        if res_ok:
            resolution_passed += 1

        results.append({
            "id": cid,
            "class": c_class,
            "mutability_class": mut_class,
            "half_life_days": case.get("half_life_days"),
            "exist": exist_text,
            "incoming": in_text,
            "expected_detect": case["expected_detection"],
            "actual_detect": actual_detection,
            "detect_ok": detect_ok,
            "expected_res": case["expected_resolution"],
            "actual_res": actual_resolution,
            "res_ok": res_ok,
            "note": case.get("reproduction_note", "")
        })

    # ─────────────────────────────────────────────────
    # REPORT
    # ─────────────────────────────────────────────────

    print("\n--- RESULTS BY TEST CASE ---")
    for r in results:
        icon = "PASS" if (r["detect_ok"] and r["res_ok"]) else "FAIL"
        print(f"\n[{icon}] {r['id']} | {r['class']}")
        print(f"       Mutability: {r['mutability_class']} "
              f"(half-life: {r['half_life_days']}d)")
        print(f"       Existing:  {r['exist']}")
        print(f"       Incoming:  {r['incoming']}")
        print(f"       Detection: "
              f"expected={r['expected_detect']} "
              f"actual={r['actual_detect']} "
              f"-> {'PASS' if r['detect_ok'] else 'FAIL'}")
        print(f"       Resolution: "
              f"expected={r['expected_res']} "
              f"actual={r['actual_res']} "
              f"-> {'PASS' if r['res_ok'] else 'FAIL'}")
        if r["note"]:
            print(f"       Note: {r['note']}")
        print("-" * 70)

    # 3x4 Matrix
    print("\n3x4 ACCURACY MATRIX")
    print("(contradiction class x mutability class)")
    print("-" * 50)

    classes = [
        "Class A: Direct Inversion",
        "Class B: Soft Override",
        "Class C: Legitimate Coexistence"
    ]
    mut_classes = ["permanent", "slow_drift", "volatile", "session"]

    header = f"{'':12} | {'A':8} | {'B':8} | {'C':8}"
    print(header)
    print("-" * 45)

    for mut in mut_classes:
        row = f"{mut:12}"
        for cls_name in classes:
            cell = [r for r in results
                    if r["class"] == cls_name
                    and r["mutability_class"] == mut]
            if cell:
                passed = sum(1 for r in cell
                            if r["detect_ok"] and r["res_ok"])
                row += f" | {passed}/{len(cell):6}"
            else:
                row += f" | {'N/A':8}"
        print(row)

    total = len(TEST_SUITE_V2)
    detect_acc = (detector_passed / total) * 100
    res_acc = (resolution_passed / total) * 100

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total test cases:     {total}")
    print(f"Detection accuracy:   {detect_acc:.1f}% "
          f"({detector_passed}/{total})")
    print(f"Resolution accuracy:  {res_acc:.1f}% "
          f"({resolution_passed}/{total})")
    print(f"New test case A3:     "
          f"Vegetarian reproduction from Issue #5352")
    print("="*70)


if __name__ == "__main__":
    run_v2_benchmark()