from huggingface_hub import login

token = input("Enter Hugging Face token: ").strip()

login(token=token)

print("Login successful")
