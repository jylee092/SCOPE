# Rule Coverage Audit (Final)

- 데이터셋: 37개 JSON
- 사용 가능한 룰: **46개**

## Aggregate
- 필요 technique 총 45개 중 **45개 커버**
- 미커버: `(없음)`

## Per-Scenario Coverage

| Group / Tactic | Scenario | 총 이벤트 | Techniques | Exact | Parent | Uncovered |
|---|---|---|---|---|---|---|
| atomic/collection | msf_record_mic | 6,202 | T1059.001,T1070.004,T1123 | T1059.001,T1070.004,T1123 | - | - |
| atomic/credential_access | cmd_dumping_ntds_dit_file_ntdsutil | 11,184 | T1003.003,T1070.001,T1070.004 | T1070.001,T1070.004 | T1003.003 | - |
| atomic/credential_access | cmd_sam_copy_esentutl | 915 | T1003.002,T1070.001 | T1070.001 | T1003.002 | - |
| atomic/credential_access | empire_mimikatz_logonpasswords | 6,026 | T1003.001,T1059.001 | T1003.001,T1059.001 | - | - |
| atomic/credential_access | empire_shell_rubeus_asktgt_createnetonly | 3,590 | T1003.001,T1059.001,T1558.003 | T1003.001,T1059.001,T1558.003 | - | - |
| atomic/credential_access | psh_lsass_memory_dump_comsvcs | 184 | T1003.001,T1070.001 | T1003.001,T1070.001 | - | - |
| atomic/defense_evasion | cmd_process_herpaderping_mimiexplorer | 265 | T1003.001,T1036.005,T1070.001 | T1003.001,T1036.005,T1070.001 | - | - |
| atomic/defense_evasion | cmd_stop_event_logging_controlset001_minint_key | 16,010 | T1070.001,T1070.004,T1562.001 | T1070.001,T1070.004,T1562.001 | - | - |
| atomic/defense_evasion | cmd_wevtutil_modify_security_eventlog_path | 25,208 | T1070.001,T1070.004 | T1070.001,T1070.004 | - | - |
| atomic/defense_evasion | empire_dllinjection_LoadLibrary_CreateRemoteThread | 12,200 | T1055,T1055.001,T1055.002,T1059.001,T1082,T1087.002 | T1055,T1059.001,T1082,T1087.002 | T1055.001,T1055.002 | - |
| atomic/defense_evasion | psh_cmstp_execution_bypassuac | 1,166 | T1070.001,T1071.001,T1218.003,T1548.002 | T1070.001,T1071.001,T1218.003,T1548.002 | - | - |
| atomic/discovery | cmd_seatbelt_group_user | 451 | T1070.001,T1082,T1087.001 | T1070.001,T1082,T1087.001 | - | - |
| atomic/discovery | empire_find_localadmin_smb_svcctl_OpenSCManager | 4,570 | T1003.001,T1033,T1059.001,T1087.002 | T1003.001,T1033,T1059.001,T1087.002 | - | - |
| atomic/discovery | empire_getsession_dcerpc_smb_srvsvc_NetSessEnum | 5,047 | T1003.001,T1049,T1059.001,T1070.004,T1087.002 | T1003.001,T1049,T1059.001,T1070.004,T1087.002 | - | - |
| atomic/discovery | empire_shell_net_local_users | 1,907 | T1003.001,T1059.001,T1087.001 | T1003.001,T1059.001,T1087.001 | - | - |
| atomic/discovery | empire_shell_samr_EnumDomainUsers | 633 | T1059.001 | T1059.001 | - | - |
| atomic/execution | cmd_sharpview_pcre_net | 267 | T1059.003,T1070.001,T1087.002 | T1059.003,T1070.001,T1087.002 | - | - |
| atomic/execution | empire_launcher_vbs | 2,067 | T1027,T1059.001,T1059.005,T1105,T1140 | T1027,T1059.001,T1059.005,T1105,T1140 | - | - |
| atomic/execution | psh_powershell_httplistener | 110 | T1059.001,T1070.001 | T1059.001,T1070.001 | - | - |
| atomic/execution | psh_python_webserver | 2,395 | T1059.006,T1070.001 | T1070.001 | T1059.006 | - |
| atomic/lateral_movement | covenant_psremoting_command | 4,284 | T1003.001,T1021.006 | T1003.001,T1021.006 | - | - |
| atomic/lateral_movement | empire_psexec_dcerpc_tcp_svcctl | 4,348 | T1003.001,T1021.002,T1027,T1059.001,T1059.003,T1105,T1140 | T1003.001,T1021.002,T1027,T1059.001,T1059.003,T1105,T1140 | - | - |
| atomic/lateral_movement | empire_psremoting_stager | 2,744 | T1003.001,T1021.006,T1027,T1059.001,T1105,T1140 | T1003.001,T1021.006,T1027,T1059.001,T1105,T1140 | - | - |
| atomic/lateral_movement | empire_wmi_dcerpc_wmi_IWbemServices_ExecMethod | 6,383 | T1003.001,T1021.003,T1027,T1047,T1059.001,T1087.002,T1105,T1140 | T1003.001,T1021.003,T1027,T1047,T1059.001,T1087.002,T1105,T1140 | - | - |
| atomic/lateral_movement | purplesharp_ad_playbook_I | 25,993 | T1021.006,T1059.001,T1069.002,T1070.004,T1087.002 | T1021.006,T1059.001,T1069.002,T1070.004,T1087.002 | - | - |
| atomic/persistence | cmd_userinitmprlogonscript_batch | 122 | T1037.001,T1070.001 | T1037.001,T1070.001 | - | - |
| atomic/persistence | covenant_persistwmi | 1,001 | T1546.003 | T1546.003 | - | - |
| atomic/persistence | empire_persistence_registry_modification_run_keys_elevated_user | 657 | T1547.001 | T1547.001 | - | - |
| atomic/persistence | empire_schtasks_creation_execution_elevated_user | 59,399 | T1003.001,T1016,T1021.006,T1027,T1053.005,T1055,T1059.001,T1059.003,T1070.004,T1087.002,T1105,T1140 | T1003.001,T1016,T1021.006,T1027,T1053.005,T1055,T1059.001,T1059.003,T1070.004,T1087.002,T1105,T1140 | - | - |
| atomic/persistence | empire_wmi_local_event_subscriptions_elevated_user | 79,896 | T1003.001,T1016,T1021.006,T1027,T1055,T1059.001,T1059.003,T1070.004,T1071.001,T1105,T1140,T1546.003 | T1003.001,T1016,T1021.006,T1027,T1055,T1059.001,T1059.003,T1070.004,T1071.001,T1105,T1140,T1546.003 | - | - |
| atomic/privilege_escalation | cmd_service_mod_fax | 437 | T1059.001,T1070.001,T1543.003 | T1059.001,T1070.001,T1543.003 | - | - |
| atomic/privilege_escalation | empire_invoke_runas | 2,581 | T1059.001,T1134 | T1059.001,T1134 | - | - |
| atomic/privilege_escalation | empire_uac_shellapi_fodhelper | 4,139 | T1027,T1059.001,T1105,T1140,T1548.002 | T1027,T1059.001,T1105,T1140,T1548.002 | - | - |
| compound/apt29 | apt29_evals_day1_manual | 100,000 | T1003.001,T1021.002,T1021.006,T1027,T1047,T1053.005,T1055,T1057,T1059.001,T1070.001,T1070.004,T1071.001,T1082,T1083,T1087.001,T1087.002,T1140,T1218.005,T1482,T1546.003,T1547.001,T1548.002,T1560.001 | T1003.001,T1021.002,T1021.006,T1027,T1047,T1053.005,T1055,T1057,T1059.001,T1070.001,T1070.004,T1071.001,T1082,T1083,T1087.001,T1087.002,T1140,T1218.005,T1482,T1546.003,T1547.001,T1548.002,T1560.001 | - | - |
| compound/apt29 | apt29_evals_day2_manual | 100,000 | T1003.001,T1018,T1021.002,T1021.006,T1027,T1047,T1053.005,T1055,T1059.001,T1070.001,T1070.004,T1071.001,T1083,T1087.001,T1087.002,T1105,T1140,T1218.005,T1482,T1546.003,T1547.001,T1548.002 | T1003.001,T1018,T1021.002,T1021.006,T1027,T1047,T1053.005,T1055,T1059.001,T1070.001,T1070.004,T1071.001,T1083,T1087.001,T1087.002,T1105,T1140,T1218.005,T1482,T1546.003,T1547.001,T1548.002 | - | - |
| compound/lsass_campaign_01 | metasploit_logonpasswords_lsass_memory_dump | 53,698 | T1003.001,T1016,T1033,T1055,T1057,T1059.001,T1059.005,T1070.001,T1070.004,T1071.001,T1082 | T1003.001,T1016,T1033,T1055,T1057,T1059.001,T1059.005,T1070.001,T1070.004,T1071.001,T1082 | - | - |
| compound/lsass_campaign_02 | metasploit_procdump_lsass_memory_dump | 42,482 | T1003.001,T1016,T1033,T1057,T1059.001,T1059.005,T1070.001,T1070.004,T1082 | T1003.001,T1016,T1033,T1057,T1059.001,T1059.005,T1070.001,T1070.004,T1082 | - | - |