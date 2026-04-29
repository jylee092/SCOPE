"""Batch 02 -- pending 2-4 scenarios."""
from experiments.apply_labels import apply_decisions

# ── empire_persistence_registry_modification_run_keys_elevated_user ──
# Both groups: SYSTEM user + 28 net conn + EID 3,4661 -- background network/handle
# events not tied to the Run-key persistence attack. The actual attack already
# captured by anchor-tool labels (powershell.exe T1547.001).
print(apply_decisions("empire_persistence_registry_modification", {
    "T1069_002_50": {"is_attack": False, "tid": None, "step": None,
                     "reason": "SYSTEM background net/handle events -- not the Run-key attack",
                     "confidence": 0.85},
    "T1087_002_42": {"is_attack": False, "tid": None, "step": None,
                     "reason": "SYSTEM background net/handle events -- not the Run-key attack",
                     "confidence": 0.85},
}))

# ── empire_shell_samr_EnumDomainUsers ──
# Both anchor on NetworkWatcherAgent.exe (Azure Network Watcher Agent doing
# routine packet capture cleanup). Unrelated to SAMR enum attack.
print(apply_decisions("empire_shell_samr_EnumDomainUsers", {
    "T1070_004_570": {"is_attack": False, "tid": None, "step": None,
                      "reason": "Azure NetworkWatcherAgent routine cleanup -- not the attack",
                      "confidence": 0.95},
    "T1112_387":     {"is_attack": False, "tid": None, "step": None,
                      "reason": "Azure NetworkWatcherAgent registry housekeeping -- not the attack",
                      "confidence": 0.95},
}))

# ── covenant_persistwmi ──
# T1047_431: EID 5861/5857/5859 are WMI operational events for binding/filter/
#   provider load. Likely Covenant WMI subscription firing → T1546.003.
# T1134_330: svchost --access--> powershell GA=0x1400 -- moderate access; EID
#   4673 priv-use; ambiguous, likely benign Windows internal probe.
# T1546_003_351: EID 19/20/21 = WMI EventFilter/EventConsumer/Binding registration.
#   This IS the WMI persistence registration -- TP for T1546.003 (step 2).
print(apply_decisions("covenant_persistwmi", {
    "T1047_431": {"is_attack": True, "tid": "T1546.003", "step": 2,
                  "reason": "EID 5861 (WMI binding consumer) -- Covenant WMI subscription firing",
                  "confidence": 0.80},
    "T1134_330": {"is_attack": False, "tid": None, "step": None,
                  "reason": "svchost moderate access on powershell -- likely Windows internal",
                  "confidence": 0.70},
    "T1546_003_351": {"is_attack": True, "tid": "T1546.003", "step": 2,
                  "reason": "EID 19/20/21 -- WMI EventFilter/EventConsumer/Binding registered",
                  "confidence": 0.95},
}))

# ── empire_mimikatz_logonpasswords ──
# T1033_281, T1033_303, T1047_532: all `whoami /user` spawned by powershell
#   (Empire runs whoami early for situational awareness). T1033 System Owner
#   Discovery -- Empire side-effect, NOT the mimikatz attack itself but is real
#   attack activity (Discovery side-step before CredAccess).
# T1112_114: backgroundTaskHost writes HKLM\System\CCS\Services\bam\State (BAM
#   service tracking app usage) -- benign Windows housekeeping.
print(apply_decisions("empire_mimikatz_logonpasswords", {
    "T1033_281": {"is_attack": True, "tid": "T1033", "step": None,
                  "reason": "whoami /user -- Empire situational awareness (Discovery side-step)",
                  "confidence": 0.90},
    "T1033_303": {"is_attack": True, "tid": "T1033", "step": None,
                  "reason": "whoami /user -- Empire situational awareness",
                  "confidence": 0.90},
    "T1047_532": {"is_attack": True, "tid": "T1033", "step": None,
                  "reason": "whoami /user spawned by powershell -- Empire discovery",
                  "confidence": 0.85},
    "T1112_114": {"is_attack": False, "tid": None, "step": None,
                  "reason": "backgroundTaskHost BAM service registry write -- benign",
                  "confidence": 0.95},
}))

# ── empire_shell_net_local_users ──
# T1021_003_441, T1021_003_1492: WmiPrvSE.exe spawned by svchost. WMI may be
#   incidentally triggered by Empire's PowerShell calls but the actual `net user`
#   command isn't here -- these are WMI provider housekeeping, no clear attack
#   signal. Benign-side-effect.
# T1047_550: sppsvc + WmiPrvSE -- Windows Software Protection + WMI background.
# T1546_015_1312: sppsvc.exe (Software Protection Service) -- benign system.
print(apply_decisions("empire_shell_net_local_users", {
    "T1021_003_441":  {"is_attack": False, "tid": None, "step": None,
                       "reason": "WmiPrvSE provider activity -- WMI side-effect, not the net.exe attack",
                       "confidence": 0.80},
    "T1021_003_1492": {"is_attack": False, "tid": None, "step": None,
                       "reason": "WmiPrvSE provider activity -- WMI side-effect",
                       "confidence": 0.80},
    "T1047_550":      {"is_attack": False, "tid": None, "step": None,
                       "reason": "sppsvc + WmiPrvSE -- Windows licensing/WMI benign",
                       "confidence": 0.90},
    "T1546_015_1312": {"is_attack": False, "tid": None, "step": None,
                       "reason": "sppsvc.exe Software Protection Service -- benign Windows component",
                       "confidence": 0.95},
}))
