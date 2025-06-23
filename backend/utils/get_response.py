from config import get_client

client = get_client()
def get_response(query, context, instructions):
    """
    Trả về phản hồi từ mô hình dựa trên truy vấn và ngữ cảnh.
    """
    # Giả lập phản hồi từ mô hình
    question = f"Câu hỏi: {query}\nDữ liệu: {context}"
    messages = [
        ("system", instructions),
        (
            "user",
            f"""Dưới đây là dữ liệu hỗ trợ:
    {context}

    Dựa vào đó, hãy trả lời câu hỏi sau:
    {query}"""
        ),
    ]

    ai_msg = client.invoke(messages)
    return ai_msg
