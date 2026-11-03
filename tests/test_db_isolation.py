import sqlite3
import pytest

from fastapi.testclient import TestClient
from api import app

def test_database_isolation(tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)

    connection = sqlite3.connect("papers.db")

    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE papers(
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           title TEXT,
           year INTEGER
        )
    """)

    cursor.execute(
        "INSERT INTO papers(title,year) VALUES(?,?)",
        ("ISOLATION_TEST",2026)
    )

    connection.commit()
    connection.close()

    assert (tmp_path / "papers.db").exists()

    client = TestClient(app)
    response = client.get("/papers")
    assert response.status_code == 200
    assert response.json()[0]["title"] == "ISOLATION_TEST"

    post_response = client.post(
        "/papers",
        json={
            "title":"PYTEST_POST",
            "year":2025
        }
    )

    assert post_response.status_code == 200
    assert post_response.json()["title"] == "PYTEST_POST"

    get_response = client.get("/papers")
    papers = get_response.json()

    assert len(papers) == 2
    assert papers[1]["title"] == "PYTEST_POST"

    #07-8：PUT 自动化测试
    paper_id = post_response.json()["id"]
    put_response = client.put(
        f"/papers/{paper_id}",
        json={"year":2026}
    )
    assert put_response.status_code == 200
    get_response = client.get(f"/papers/{paper_id}")
    assert get_response.status_code == 200
    assert get_response.json()["year"] == 2026

    #07-9｜DELETE 自动化测试
    delete_response = client.delete(
        f"/papers/{paper_id}"
    )
    assert delete_response.status_code == 200
    get_response = client.get("/papers")
    papers = get_response.json()
    assert get_response.status_code == 200
    assert all(paper["id"] != paper_id for paper in papers)

    #07-10｜测试重复删除返回 HTTP 404
    delete_response2 = client.delete(
        f"/papers/{paper_id}"
    )
    assert delete_response2.status_code == 404
    assert delete_response2.json()["detail"] == "论文不存在"




    #08｜pytest fixture：让测试互不干扰
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    connection = sqlite3.connect("papers.db")

    cursor = connection.cursor()

    cursor.execute("""
            CREATE TABLE papers(
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               title TEXT,
               year INTEGER
            )
        """)

    connection.commit()
    connection.close()

    yield TestClient(app)

def test_empty_papers(client):
    response = client.get("/papers")
    assert response.status_code == 200
    assert response.json() == []


    #08-2｜独立编写 POST 自动化测试
def test_create_paper(client):
    response = client.post("/papers",json={
        "title":"PYTEST_CREATE",
        "year":2026
    })
    assert response.status_code == 200
    assert response.json()["title"] == "PYTEST_CREATE"
    assert response.json()["year"] == 2026

    paper_id = response.json()["id"]
    get_response = client.get(f"/papers/{paper_id}")
    assert get_response.status_code == 200
    assert get_response.json()["title"] == "PYTEST_CREATE"
    assert get_response.json()["year"] == 2026


    #08-3｜独立 PUT 自动化测试
def test_update_paper(client):
    post_response = client.post("/papers",json={
        "title":"PUT_TEST",
        "year":2024
    })
    assert post_response.status_code == 200
    paper_id = post_response.json()["id"]
    put_response = client.put(f"/papers/{paper_id}",json={"year":2026})
    assert put_response.status_code == 200
    get_response = client.get(f"/papers/{paper_id}")
    assert get_response.status_code == 200
    assert get_response.json()['year'] == 2026
    assert get_response.json()["title"] == "PUT_TEST"

    #08-4｜独立 DELETE 自动化测试
def test_delete_paper(client):
    post_response = client.post("/papers",json={
        "title":"DELETE_TEST",
        "year":2026
    })
    assert post_response.status_code == 200
    paper_id = post_response.json()["id"]
    delete_response = client.delete(f"/papers/{paper_id}")
    assert delete_response.status_code == 200
    get_response = client.get(f"/papers")
    papers = get_response.json()
    assert get_response.status_code == 200
    assert all(paper["id"] != paper_id for paper in papers)

    delete_response2 = client.delete(
            f"/papers/{paper_id}"
        )
    assert delete_response2.status_code == 404
    assert delete_response2.json()["detail"] == "论文不存在"
    