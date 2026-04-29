"""Batch 01 -- small-pending scenarios (1-2 groups each).

"""
from experiments.apply_labels import apply_decisions

results = []

# ── cmd_seatbelt_group_user ──
# T1546_015_306: WmiPrvSE.exe spawned by svchost -- OS WMI provider responding
# to (likely) Seatbelt's WMI query. Anchor process is system component, not the
# attacker tool. Rule fired T1546.015 (COM hijacking) but no COM hijack here.
results.append(apply_decisions("cmd_seatbelt_group_user", {
    "T1546_015_306": {
        "is_attack": False, "tid": None, "step": None,
        "reason": "WMI provider svchost spawn -- OS internal, not COM hijacking",
        "confidence": 0.85,
    },
}))

# ── psh_powershell_httplistener ──
# T1055_0: svchost --access--> backgroundTaskHost with GA=0x1000 (low) → benign.
# Not the PowerShell HTTP listener attack itself.
results.append(apply_decisions("psh_powershell_httplistener", {
    "T1055_0": {
        "is_attack": False, "tid": None, "step": None,
        "reason": "svchost low-GA(0x1000) probe on backgroundTaskHost -- benign",
        "confidence": 0.90,
    },
}))

# ── cmd_userinitmprlogonscript_batch ──
# T1037_001_75: REG.exe ADD HKCU\Environment\UserInitMprLogonScript=art.bat
# → exact T1037.001 Logon Script persistence. Step 2 in flow.
results.append(apply_decisions("cmd_userinitmprlogonscript", {
    "T1037_001_75": {
        "is_attack": True, "tid": "T1037.001", "step": 2,
        "reason": r"reg add HKCU\Environment\UserInitMprLogonScript=art.bat",
        "confidence": 0.98,
    },
}))

# ── psh_python_webserver ──
# T1003_2220: svchost --access--> lsass with GA=0x2000 (PROCESS_DUP_HANDLE).
# Not a credential dump (would need 0x1010/0x1410). Benign Windows internal.
results.append(apply_decisions("psh_python_webserver", {
    "T1003_2220": {
        "is_attack": False, "tid": None, "step": None,
        "reason": "svchost GA=0x2000 (DUP_HANDLE) on lsass -- benign Windows internal",
        "confidence": 0.90,
    },
}))

# ── cmd_service_mod_fax ──
# T1112_36: services.exe registry write to HKLM\System\CCS\Services\Fax\ImagePath
# = powershell payload. THIS IS the T1543.003 attack effect. Step 2 in flow.
results.append(apply_decisions("cmd_service_mod_fax", {
    "T1112_36": {
        "is_attack": True, "tid": "T1543.003", "step": 2,
        "reason": r"services.exe writes Services\Fax\ImagePath = powershell payload",
        "confidence": 0.97,
    },
}))

for r in results:
    print(r)
