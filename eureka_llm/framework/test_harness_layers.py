from context_packet import build_evidence_packet
from prompt_harness import build_contract_block
from runtime_policy import compute_evidence_fingerprint, should_rerun_analyst


def run_tests():
    packet = build_evidence_packet(
        diagnostics={"mean_length": 120},
        must_fix_issues=["fix A", "fix B", "fix C", "fix D"],
        mandatory_lessons=["lesson A", "lesson B"],
        strategy="strict",
    )
    assert "Evidence Packet" in packet
    assert "fix D" not in packet

    contract = build_contract_block("Analyst", "obj", ["out1"], ["hard1"])
    assert "Prompt Contract" in contract

    fp1 = compute_evidence_fingerprint(["x"], {"changed_count": 1, "proposed_changes": []})
    fp2 = compute_evidence_fingerprint(["x"], {"changed_count": 1, "proposed_changes": []})
    assert fp1 == fp2
    assert not should_rerun_analyst(fp1, fp2, strategy="strict")
    assert should_rerun_analyst(fp1, fp2, strategy="always_once")


if __name__ == "__main__":
    run_tests()
    print("ok")
