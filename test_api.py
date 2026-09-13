import requests

BASE_URL = "http://127.0.0.1:8000"

def test_root_endpoint():
    print("[1/3] Testing Root Endpoint...")
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    print("  ✅ Status Code 200 OK")
    print("  Response:", response.json())

def test_analytics_summary():
    print("\n[2/3] Testing Analytics Summary Endpoint...")
    response = requests.get(f"{BASE_URL}/api/v1/analytics/summary")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert "total_feedback" in data, "Key 'total_feedback' missing"
    assert "average_rating" in data, "Key 'average_rating' missing"
    print("  ✅ Status Code 200 OK")
    print(f"  Summary: Total Feedback = {data.get('total_feedback')}, Avg Rating = {data.get('average_rating')}")

def test_rag_ask():
    print("\n[3/3] Testing RAG Ask Endpoint...")
    payload = {"question": "Apa saja keluhan utama pelanggan terkait sistem pembayaran?"}
    response = requests.post(f"{BASE_URL}/api/v1/rag/ask", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert "answer" in data, "Key 'answer' missing in response"
    print("  ✅ Status Code 200 OK")
    print("  Q:", payload["question"])
    print("  A:", data["answer"])

if __name__ == "__main__":
    print("=" * 50)
    print(" STARTING AUTOMATED API TESTS")
    print("=" * 50)
    try:
        test_root_endpoint()
        test_analytics_summary()
        test_rag_ask()
        print("\n" + "=" * 50)
        print(" 🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 50)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except requests.exceptions.ConnectionError:
        print("\n❌ CONNECTION ERROR: Pastikan Uvicorn server sudah berjalan di http://127.0.0.1:8000")