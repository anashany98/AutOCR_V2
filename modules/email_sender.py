import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailSender:
    def __init__(self, config):
        self.logger = logging.getLogger("EmailSender")
        self.config = config.get("email_importer", {}) # Reusing email config or create new section?
        # Let's check for specific SMTP config, fallback to importer if needed/possible but usually different
        self.smtp_host = self.config.get("smtp_host")
        self.smtp_port = int(self.config.get("smtp_port", 587))
        self.smtp_user = self.config.get("smtp_user") # Often same as importer user
        self.smtp_password = self.config.get("smtp_password") # Often same
        self.smtp_tls = self.config.get("smtp_tls", True)
        self.sender_email = self.config.get("sender_email", self.smtp_user)

    def send_email(self, to_email, subject, body_html):
        if not self.smtp_host or not self.smtp_user:
            self.logger.warning("SMTP Config missing. Email not sent.")
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = to_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body_html, 'html'))

            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.ehlo()
            if self.smtp_tls:
                server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.sendmail(self.sender_email, to_email, msg.as_string())
            server.close()
            self.logger.info(f"Email sent to {to_email}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False

    def send_verification_email(self, to_email, token, base_url):
        link = f"{base_url}/auth/verify/{token}"
        subject = "Verifica tu cuenta en AutoOCR"
        body = f"""
        <h2>Bienvenido a AutoOCR</h2>
        <p>Gracias por registrarte. Para activar tu cuenta, por favor verifica tu correo electrónico:</p>
        <p><a href="{link}" style="padding: 10px 20px; background-color: #0d6efd; color: white; text-decoration: none; border-radius: 5px;">Verificar Cuenta</a></p>
        <p>O copia este enlace: {link}</p>
        """
        return self.send_email(to_email, subject, body)

    def send_reset_email(self, to_email, token, base_url):
        link = f"{base_url}/auth/reset-password/{token}"
        subject = "Recuperar Contraseña - AutoOCR"
        body = f"""
        <h2>Recuperación de Contraseña</h2>
        <p>Has solicitado restablecer tu contraseña. Haz clic en el siguiente enlace:</p>
        <p><a href="{link}" style="padding: 10px 20px; background-color: #dc3545; color: white; text-decoration: none; border-radius: 5px;">Restablecer Contraseña</a></p>
        <p>Este enlace expirará en 1 hora.</p>
        """
        return self.send_email(to_email, subject, body)
