# Test Duration Report

- Source: `tmp\full-junit-optimized.xml`
- Total test time (sum of testcase times): `0.865s`

## By File

| File | Seconds |
|---|---:|
| `tests/test_auth_security_regressions.py` | 0.355 |
| `tests/test_vision_manager.py` | 0.292 |
| `tests/test_pdf_chunking.py` | 0.060 |
| `tests/test_chat_security.py` | 0.052 |
| `tests/test_tool_manager_scoping.py` | 0.046 |
| `tests/test_pipeline.py` | 0.030 |
| `tests/test_field_quality_baseline.py` | 0.022 |
| `tests/test_table_manager.py` | 0.005 |
| `tests/test_field_extraction_core.py` | 0.002 |
| `tests/test_process_blocks.py` | 0.001 |
| `tests/test_config_normalizer.py` | 0.000 |
| `tests/test_fusion_manager.py` | 0.000 |

## Top 10 Slow Tests

| Test | File | Seconds |
|---|---|---:|
| `tests.test_vision_manager::test_build_and_search_roundtrip` | `tests/test_vision_manager.py` | 0.292 |
| `tests.test_auth_security_regressions::test_public_register_cannot_escalate_role_and_requires_verified_email` | `tests/test_auth_security_regressions.py` | 0.193 |
| `tests.test_auth_security_regressions::test_user_role_and_scope_updates_work_in_sqlite` | `tests/test_auth_security_regressions.py` | 0.073 |
| `tests.test_pdf_chunking::test_workflow_pending_when_visual_text_missing` | `tests/test_pdf_chunking.py` | 0.037 |
| `tests.test_chat_security::test_chat_history_scoped_by_user_id` | `tests/test_chat_security.py` | 0.030 |
| `tests.test_auth_security_regressions::test_get_document_path_works_with_legacy_path_column` | `tests/test_auth_security_regressions.py` | 0.026 |
| `tests.test_pdf_chunking::test_pdf_chunking_uses_ranges_and_global_pages` | `tests/test_pdf_chunking.py` | 0.023 |
| `tests.test_tool_manager_scoping::test_tool_export_respects_hotel_scope` | `tests/test_tool_manager_scoping.py` | 0.016 |
| `tests.test_tool_manager_scoping::test_tool_update_document_type_scoped` | `tests/test_tool_manager_scoping.py` | 0.015 |
| `tests.test_tool_manager_scoping::test_client_cannot_execute_document_mutation_tools` | `tests/test_tool_manager_scoping.py` | 0.015 |
