from dotenv import load_dotenv
import os

# cargar .env
load_dotenv()

print("EMAIL_USER:", os.getenv("EMAIL_USER"))
print("EMAIL_PASS:", os.getenv("EMAIL_PASS"))
print("SMTP_HOST:", os.getenv("SMTP_HOST"))
print("SMTP_PORT:", os.getenv("SMTP_PORT"))
