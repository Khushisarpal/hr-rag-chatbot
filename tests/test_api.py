import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def ask(q):
    return client.post("/chat", json={"question": q})


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_index_served():
    assert client.get("/").status_code == 200


@pytest.mark.parametrize("question,file,keyword", [
    ("Combien de jours de congés payés par an ?", "conges.md", "30"),
    ("Combien de jours de télétravail par semaine ?", "teletravail.md", "3 jours"),
    ("Quel est le plafond pour une nuit d'hôtel à Paris ?", "notes-de-frais.md", "180"),
    ("Quel est le budget formation annuel ?", "onboarding-formation.md", "1 500"),
    ("Quelle est la durée de la période d'essai pour un cadre ?", "onboarding-formation.md", "3 mois"),
])
def test_grounded_answers(question, file, keyword):
    body = ask(question).json()
    assert body["grounded"] is True
    assert body["sources"][0]["file"] == file
    assert keyword in body["answer"]


def test_out_of_scope_not_hallucinated():
    body = ask("Quelle est la capitale de l'Australie ?").json()
    assert body["grounded"] is False
    assert body["sources"] == []


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "a"}, {"question": "x" * 501}])
def test_validation_errors(payload):
    assert client.post("/chat", json=payload).status_code == 422
