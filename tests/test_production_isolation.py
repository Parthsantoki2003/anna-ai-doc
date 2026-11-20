import io
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch
import fitz
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main


def make_pdf_bytes(text_pages: list[str]) -> bytes:
    """Create a valid in-memory PDF with the specified text pages."""
    doc = fitz.open()
    for page_text in text_pages:
        page = doc.new_page()
        page.insert_text((72, 72), page_text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def reset_isolated_state():
    """Provide a clean in-memory Qdrant store and reset caches for every test."""
    main.qdrant = QdrantClient(":memory:")
    main.s3_client = None
    main.init_qdrant_collections()
    main.user_doc_lookups.clear()
    main.conversations.clear()
    main.workflow_cache.clear()
    main.rate_limit_buckets.clear()


def test_missing_or_weak_user_id_rejected():
    """F-01: Requests without X-User-Id or with low-entropy/default_user IDs must return 401."""
    reset_isolated_state()
    client = TestClient(main.app)
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-User-Id": "default_user"}).status_code == 401
    assert client.get("/documents", headers={"X-User-Id": "123"}).status_code == 401
    assert client.post("/query", json={"query": "hello"}, headers={"X-User-Id": "admin"}).status_code == 401


def test_cross_tenant_document_and_conversation_isolation():
    """F-01 & F-02: User B cannot list, query, analyze, delete, or read conversations belonging to User A."""
    reset_isolated_state()
    client = TestClient(main.app)
    user_a = "usr_tenant_alpha_998877"
    user_b = "usr_tenant_beta_112233"

    pdf_a = make_pdf_bytes(["Project Falcon Confidential Merger Budget is $42.5 Million for Q4 2026."])
    res_ing = client.post(
        "/ingest",
        files={"file": ("falcon_secret.pdf", io.BytesIO(pdf_a), "application/pdf")},
        headers={"X-User-Id": user_a},
    )
    assert res_ing.status_code == 200
    file_id_a = res_ing.json()["file_id"]

    # User B lists documents -> must be empty
    docs_b = client.get("/documents", headers={"X-User-Id": user_b}).json()
    assert docs_b == []

    # User B tries to target User A's file_id in /query -> must return 403
    q_forbid = client.post(
        "/query",
        json={"query": "What is the Project Falcon budget?", "file_ids": [file_id_a]},
        headers={"X-User-Id": user_b},
    )
    assert q_forbid.status_code == 403

    # User B uploads their own file and tries to target User A's file_id in /workflow/executive-summary -> must return 403
    pdf_b = make_pdf_bytes(["User B public grocery list: apples, bananas, oranges."])
    client.post(
        "/ingest",
        files={"file": ("groceries.pdf", io.BytesIO(pdf_b), "application/pdf")},
        headers={"X-User-Id": user_b},
    )
    wf_forbid = client.post(
        "/workflow/executive-summary",
        json={"file_ids": [file_id_a]},
        headers={"X-User-Id": user_b},
    )
    assert wf_forbid.status_code == 403

    # User B tries to delete User A's document -> must return 404
    del_res = client.delete(f"/documents/{file_id_a}", headers={"X-User-Id": user_b})
    assert del_res.status_code == 404

    # User A runs a query and creates a conversation
    def fake_groq_create(**kwargs):
        payload = json.dumps({
            "answer_found": True,
            "answer": "The Project Falcon merger budget is $42.5 Million.",
            "cited_chunk_ids": [1]
        })
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload))], usage=None)

    with patch.object(main.groq_client.chat.completions, "create", side_effect=fake_groq_create):
        q_a = client.post(
            "/query",
            json={"query": "What is the Project Falcon merger budget?"},
            headers={"X-User-Id": user_a},
        )
        assert q_a.status_code == 200
        conv_id_a = q_a.json()["conversation_id"]

    # User A can fetch their own conversation
    assert client.get(f"/conversations/{conv_id_a}", headers={"X-User-Id": user_a}).status_code == 200
    # User B cannot fetch User A's conversation
    assert client.get(f"/conversations/{conv_id_a}", headers={"X-User-Id": user_b}).status_code == 404


def test_unrelated_documents_citation_precision():
    """Mandatory Fix 2: Querying across two unrelated documents cites ONLY the supporting document."""
    reset_isolated_state()
    client = TestClient(main.app)
    user_id = "usr_citation_tester_001"

    pdf_k8s = make_pdf_bytes([
        "Kubernetes cluster autoscaler configuration requires setting max-node-provision-time to 15m "
        "and configuring PodDisruptionBudgets for stateful microservices in production namespaces."
    ])
    pdf_cake = make_pdf_bytes([
        "Grandma's Chocolate Fudge Cake Recipe: Whisk 85 grams of Dutch-process cocoa powder with "
        "200 grams of granulated sugar, 3 large eggs, and 120 ml of warm buttermilk."
    ])

    r1 = client.post(
        "/ingest",
        files={"file": ("kubernetes_runbook.pdf", io.BytesIO(pdf_k8s), "application/pdf")},
        headers={"X-User-Id": user_id},
    )
    r2 = client.post(
        "/ingest",
        files={"file": ("chocolate_cake_recipe.pdf", io.BytesIO(pdf_cake), "application/pdf")},
        headers={"X-User-Id": user_id},
    )
    assert r1.status_code == 200 and r2.status_code == 200

    def fake_groq_create(**kwargs):
        payload = json.dumps({
            "answer_found": True,
            "answer": "The recipe requires 85 grams of Dutch-process cocoa powder (chocolate_cake_recipe.pdf).",
            "cited_chunk_ids": [1]
        })
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload))], usage=None)

    with patch.object(main.groq_client.chat.completions, "create", side_effect=fake_groq_create):
        q_res = client.post(
            "/query",
            json={"query": "How many grams of cocoa powder are needed for the chocolate cake?"},
            headers={"X-User-Id": user_id},
        )
        assert q_res.status_code == 200
        body = q_res.json()
        cited_files = [s["source"] for s in body["sources"]]
        assert cited_files == ["chocolate_cake_recipe.pdf"]
        assert "kubernetes_runbook.pdf" not in cited_files


def test_unanswerable_query_returns_zero_citations():
    """Mandatory Fix 2: Unanswerable queries return sources=[] and grounding_score=0."""
    reset_isolated_state()
    client = TestClient(main.app)
    user_id = "usr_unanswerable_tester"
    pdf_cake = make_pdf_bytes([
        "Grandma's Chocolate Fudge Cake Recipe: Whisk 85 grams of Dutch-process cocoa powder with sugar."
    ])
    client.post(
        "/ingest",
        files={"file": ("chocolate_cake_recipe.pdf", io.BytesIO(pdf_cake), "application/pdf")},
        headers={"X-User-Id": user_id},
    )

    def fake_groq_not_found(**kwargs):
        payload = json.dumps({
            "answer_found": False,
            "answer": "Not found in the selected documents.",
            "cited_chunk_ids": []
        })
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload))], usage=None)

    with patch.object(main.groq_client.chat.completions, "create", side_effect=fake_groq_not_found):
        q_res = client.post(
            "/query",
            json={"query": "What is the orbital velocity of Jupiter's moon Europa?"},
            headers={"X-User-Id": user_id},
        )
        assert q_res.status_code == 200
        body = q_res.json()
        assert body["sources"] == []
        assert body["grounding_score"] == 0
        assert "not found in the selected documents" in body["answer"].lower()


def test_workflow_document_targeting_isolation():
    """Mandatory Fix 3: Workflows require selection when >1 docs exist and strictly isolate to selected file_ids."""
    reset_isolated_state()
    client = TestClient(main.app)
    user_id = "usr_workflow_target_001"
    pdf_x = make_pdf_bytes(["Document X covers Alpha Solar Panel Warranty Terms and 25-year inverter coverage."])
    pdf_y = make_pdf_bytes(["Document Y covers Beta Maritime Shipping Charter and demurrage penalty rates."])

    rx = client.post(
        "/ingest",
        files={"file": ("doc_X_solar.pdf", io.BytesIO(pdf_x), "application/pdf")},
        headers={"X-User-Id": user_id},
    )
    ry = client.post(
        "/ingest",
        files={"file": ("doc_Y_maritime.pdf", io.BytesIO(pdf_y), "application/pdf")},
        headers={"X-User-Id": user_id},
    )
    fid_x = rx.json()["file_id"]

    # 1. Calling Brief without selecting a document when 2 are uploaded must return 400
    res_unscoped = client.post("/workflow/executive-summary", json={}, headers={"X-User-Id": user_id})
    assert res_unscoped.status_code == 400
    assert "Multiple documents are uploaded" in res_unscoped.json()["detail"]

    # 2. Targeting doc_X_solar.pdf must ONLY send doc_X_solar.pdf chunks to the LLM and return sources=["doc_X_solar.pdf"]
    captured_contexts = []

    def fake_groq_summary(**kwargs):
        user_prompt = kwargs["messages"][1]["content"]
        captured_contexts.append(user_prompt)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "title": "Alpha Solar Warranty",
                "executive_summary": "Covers 25-year solar inverter warranty.",
                "key_findings": ["25-year coverage"],
                "critical_decisions": [],
                "timeline_or_milestones": [],
                "action_items": []
            })))],
            usage=None
        )

    with patch.object(main.groq_client.chat.completions, "create", side_effect=fake_groq_summary):
        res_x = client.post(
            "/workflow/executive-summary",
            json={"file_ids": [fid_x], "scope": "single"},
            headers={"X-User-Id": user_id},
        )
        assert res_x.status_code == 200
        data_x = res_x.json()
        assert data_x["sources"] == ["doc_X_solar.pdf"]
        assert len(captured_contexts) == 1
        assert "Alpha Solar Panel" in captured_contexts[0]
        assert "Maritime Shipping Charter" not in captured_contexts[0]


def test_anna_llm_fallback_on_all_5_workflows_and_query():
    """Verify Groq rate-limit triggers fallback_required=True with full schema prompts across all 5 workflows + query."""
    reset_isolated_state()
    client = TestClient(main.app)
    user_id = "usr_fallback_tester_007"
    pdf_1 = make_pdf_bytes(["Contract A specifies a $100,000 liability cap and 30-day termination notice."])
    pdf_2 = make_pdf_bytes(["Contract B specifies a $250,000 liability cap and 60-day termination notice."])

    r1 = client.post(
        "/ingest",
        files={"file": ("contract_A.pdf", io.BytesIO(pdf_1), "application/pdf")},
        headers={"X-User-Id": user_id},
    )
    r2 = client.post(
        "/ingest",
        files={"file": ("contract_B.pdf", io.BytesIO(pdf_2), "application/pdf")},
        headers={"X-User-Id": user_id},
    )
    fid1 = r1.json()["file_id"]
    fid2 = r2.json()["file_id"]

    def raise_rate_limit(**kwargs):
        raise RuntimeError("429 Too Many Requests: Groq rate limit reached")

    with patch.object(main.groq_client.chat.completions, "create", side_effect=raise_rate_limit):
        # 1. Query fallback
        q_res = client.post(
            "/query",
            json={"query": "What is the liability cap in Contract A?", "file_ids": [fid1]},
            headers={"X-User-Id": user_id},
        ).json()
        assert q_res["fallback_required"] is True
        assert q_res["fallback_system_prompt"] and "Not found in the selected documents" in q_res["fallback_system_prompt"]
        assert "Contract A specifies a $100,000 liability cap" in q_res["fallback_user_prompt"]

        # 2. Executive Summary fallback
        s_res = client.post(
            "/workflow/executive-summary",
            json={"file_ids": [fid1], "scope": "single"},
            headers={"X-User-Id": user_id},
        ).json()
        assert s_res["fallback_required"] is True
        assert "key_findings" in s_res["fallback_system_prompt"]

        # 3. Risk Audit fallback
        rk_res = client.post(
            "/workflow/risk-audit",
            json={"file_ids": [fid1], "scope": "single"},
            headers={"X-User-Id": user_id},
        ).json()
        assert rk_res["fallback_required"] is True
        assert "overall_risk_level" in rk_res["fallback_system_prompt"]

        # 4. Extract Data fallback
        ex_res = client.post(
            "/workflow/extract-data",
            json={"file_ids": [fid1], "scope": "single"},
            headers={"X-User-Id": user_id},
        ).json()
        assert ex_res["fallback_required"] is True
        assert "data_points" in ex_res["fallback_system_prompt"]

        # 5. Compare Documents fallback
        cmp_res = client.post(
            "/workflow/compare",
            json={"file_id_1": fid1, "file_id_2": fid2},
            headers={"X-User-Id": user_id},
        ).json()
        assert cmp_res["fallback_required"] is True
        assert "deltas" in cmp_res["fallback_system_prompt"]

        # 6. Quiz fallback
        qz_res = client.post(
            "/workflow/quiz",
            json={"file_ids": [fid1], "scope": "single", "count": 5},
            headers={"X-User-Id": user_id},
        ).json()
        assert qz_res["fallback_required"] is True
        assert "questions" in qz_res["fallback_system_prompt"]


if __name__ == "__main__":
    tests = [
        test_missing_or_weak_user_id_rejected,
        test_cross_tenant_document_and_conversation_isolation,
        test_unrelated_documents_citation_precision,
        test_unanswerable_query_returns_zero_citations,
        test_workflow_document_targeting_isolation,
        test_anna_llm_fallback_on_all_5_workflows_and_query,
    ]
    for t in tests:
        print(f"Running {t.__name__}...", end=" ", flush=True)
        t()
        print("PASSED")
    print(f"\nAll {len(tests)} production isolation & citation tests PASSED!")
