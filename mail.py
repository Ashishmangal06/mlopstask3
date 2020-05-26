import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

fromaddr="ashishmangal925@gmail.com"
toaddr="ashishmangal925@gmail.com"

msg = MIMEMultipart()

msg['From']=fromaddr

msg['To']=toaddr

msg['Subject']="Model pass or failed"

body='''<h1>hey</h1><br><br>
            '''
msg.attach(MIMEText(body, 'html', 'utf-8'))

filename="Readme.md"
attachment=open("/Readme.md","rb")

p=MIMEBase('application', 'octet-stream')

p.set_payload((attachment).read())
encoders.encode_base64(p)

p.add_header('Content-Disposition', "attachment; filename=%s" % filename)

msg.attach(p)

s=smtlib.SMTP('smtp.gmail.com', 587)

s.starttls()

s.login(fromaddr, "")
print("login successful")

text=msg.as_string()

s.sendmail(fromaddr, toaddr, text)

print("send successful")

s.quit()
