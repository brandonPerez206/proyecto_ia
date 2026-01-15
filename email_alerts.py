import smtplib
import ssl
from email.message import EmailMessage
from dotenv import load_dotenv
import os

load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT"))



def enviar_alerta(asunto, mensaje, destinatario):
    try:
        email = EmailMessage()
        email["From"] = EMAIL_USER
        email["To"] = destinatario
        email["Subject"] = asunto
        email.set_content(mensaje)

        contexto = ssl.create_default_context()

        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=contexto) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(email)

        print("Mensaje enviado correctamente")

    except Exception as e:
        print("Error enviando correo:", e)

# ---- PRUEBA ----
if __name__ == "__main__":
    enviar_alerta(
        "Prueba desde email_alerts.py",
        "Esto es una prueba de envío.",
        EMAIL_USER  # te envías el correo a ti mismo
    )
