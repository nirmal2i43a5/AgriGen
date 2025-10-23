import base64
from typing import Optional
from backend.src.llm.groq_model import get_groq_client


def process_image_question(image_bytes, question, vision_model="meta-llama/llama-4-maverick-17b-128e-instruct"):

    try:
        client = get_groq_client()
        
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_image}"
        
        print(f" Processing image question with {vision_model}...")
        
        response = client.chat.completions.create(
            model=vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            temperature=0.5,
            max_tokens=1024
        )
        
        answer = response.choices[0].message.content
        print(f" Vision response: {len(answer)} characters")
        
        return answer
    
    except Exception as e:
        print(f" Image processing failed: {e}")
        raise


def validate_image(image_bytes, max_size_mb=10):
    size_mb = len(image_bytes) / (1024 * 1024)
    
    if size_mb > max_size_mb:
        raise ValueError(f"Image too large: {size_mb:.2f}MB (max {max_size_mb}MB)")
    
    return True

