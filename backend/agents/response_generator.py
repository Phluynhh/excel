from utils.get_response import get_response


class ResponseGenerator:
    def __init__(self, model):
        self.model = model

    def is_short_greeting(self, query):
        greetings = ["hi", "hello", "xin chào", "chào", "hey", "good morning"]
        return query.lower().strip() in greetings

    def is_pure_emotion(self, query):
        # Câu chỉ chứa biểu cảm / cảm xúc ngắn gọn
        emotion_phrases = [
            "tôi buồn", "tôi mệt", "tôi vui", "tôi stress", "tôi chán",
            "tôi xinh đẹp", "tôi thông minh", "huhu", "haha", "ôi", "trời ơi"
        ]
        return query.lower().strip() in emotion_phrases or len(query.strip().split()) <= 5 and any(e in query.lower() for e in emotion_phrases)

    def generate_response(self, query, context):
        try:
            query = query.strip()

            if self.is_short_greeting(query):
                return "Chào bạn, mình có thể giúp gì?"

            if self.is_pure_emotion(query):
                return "Mình hiểu cảm xúc của bạn. Có gì mình có thể giúp không?"

            # Gọi mô hình nếu không phải 2 trường hợp trên
            print("="*20)
            print("Query:", query)
            print("Actual context passed to model:\n", context[:1500])
            print("="*20)
            print("context", context)

            instructions = """
Bạn là một chuyên viên trả lời câu hỏi dựa trên các dữ liệu được cung cấp.
Nếu người dùng bày tỏ cảm xúc, trạng thái, hãy phản hồi lại một cách phù hợp và thân thiện.
Chỉ sử dụng dữ liệu bên dưới để trả lời. Nếu thông tin có liên quan gián tiếp, bạn vẫn có thể rút ra câu trả lời hợp lý từ đó.
Nếu không có bất kỳ thông tin nào liên quan, hãy trả lời: "Tôi không tìm thấy thông tin trong tài liệu."

Trả lời ngắn gọn và đúng trọng tâm.
"""
            return get_response(query, context, instructions).content

        except Exception as e:
            print(f"Error: {str(e)}")
            raise ValueError(f"Error while generating response: {str(e)}")
