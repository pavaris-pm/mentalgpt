# Mentalgpt 👩🏻‍⚕️
 This is a project aimed at developing a voice-based chatbot that provides mental health recommendations to users who are feeling sad or depressed. The project is designed to enable users to communicate with the system using natural language, and receive personalized audio responses that provide emotional support and guidance.
 
 
## Simple Usage

```python
from transformers import pipeline

# simply calling model in just only one line using pipeline from huggingface
mental_model = pipeline('text-generation','tontokoton/mentalgpt-gpt2')

# adjust parameters for text generation
output = mental_model(f"<|startoftext|>Q: มี วิธี ไหน ที่ จะ ช่วย ลด ความ เครียด ได้ บ้าง ไหม ครับ?\n\nA:  ", 
                      top_k=50, 
                      do_sample=True,
                      num_beams=5, 
                      no_repeat_ngram_size=2, 
                      early_stopping=True, 
                      max_length=150, 
                      top_p=0.95, 
                      temperature=1.9, 
                      num_return_sequences=1)
```
```
output: 
Q: มีวิธีไหนที่จะช่วยลดความเครียดได้บ้างไหมครับ?
A: การคลายความเครียดเพื่อให้อารมณ์เย็นลงได้นั้นไม่สามารถทำให้เกิดความสงบหรือไม่ดีต่อชีวิตผู้อื่นได้ แต่เป็นเรื่องที่ควรทำเมื่อมีความตระหนักว่าสิ่งต่างๆเหล่านี้อาจมีผลต่อสิ่งแวดล้อมและสังคมโดยรวมของเรา 
```

The Mental Health Chatbot uses a combination of natural language processing and sentiment analysis technologies to detect the user's emotional state and provide appropriate recommendations. The system is designed to learn and adapt to the user's specific emotional patterns, enabling it to provide customized and effective support.

The chatbot is built on a Python-based platform, and utilizes speech recognition libraries such as PythaiNLP, and text-to-speech libraries such as KhanomtanTTS for generating audio responses. The project is a collaboration between engineers who are passionate about developing innovative voice-based solutions for improving mental health and well-being.

The Mental Health Chatbot is intended to provide users with a safe and confidential way to seek emotional support and guidance, and is not intended to replace professional mental health services. Users are encouraged to seek professional help if they are experiencing severe or persistent symptoms of depression or other mental health conditions.


