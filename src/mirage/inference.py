import base64

from openai import OpenAI


RAW_PROMPT = r"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```\nThought: ...
Action: ...\n```

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')
type(content='')
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
press_home()
press_back()
finished(content='') # Submit the task regardless of whether it succeeds or fails.

## Note
- Use English in `Thought` part.

- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
"""

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:9555/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
image_path = "data/AITW/process/google_apps/6252617148909038128/6252617148909038128_2.png"
with open(image_path, "rb") as f:
    encoded_image = base64.b64encode(f.read())
encoded_image_text = encoded_image.decode("utf-8")
base64_qwen = f"data:image;base64,{encoded_image_text}"

chat_response = client.chat.completions.create(
    model="models/UI-TARS-2B-SFT",
    messages=[
        {"role": "system", "content": RAW_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": base64_qwen},
                },
                {
                    "type": "text",
                    "text": "<image> The current user goal is: turn off location\n\n'What's the next action?",
                },
            ],
        },
    ],
)
# print("Chat response:", chat_response)
print(chat_response.choices[0].message.content)
