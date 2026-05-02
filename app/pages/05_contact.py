import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import resend
import streamlit as st

st.title("Contact")
st.caption("Questions, feedback, or just want to say hi — send me a message.")

with st.form("contact_form"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message", height=180)
    submitted = st.form_submit_button("Send message")

if submitted:
    if not message.strip():
        st.error("Please enter a message.")
    else:
        resend.api_key = st.secrets["RESEND_API_KEY"]

        subject = f"PaceCurve message{f' from {name}' if name.strip() else ''}"
        body_parts = []
        if name.strip():
            body_parts.append(f"<strong>{name.strip()}</strong>")
        if email.strip():
            body_parts.append(email.strip())
        body_parts.append(f"<p style='white-space:pre-wrap'>{message.strip()}</p>")

        params: resend.Emails.SendParams = {
            "from": st.secrets["EMAIL_FROM"],
            "to": [st.secrets["EMAIL_TO"]],
            "reply_to": email.strip() or None,
            "subject": subject,
            "html": "<br>".join(body_parts),
        }

        try:
            resend.Emails.send(params)
            st.success("Message sent — I'll get back to you soon.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
