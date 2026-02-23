import re

with open("app.py", "r") as f:
    code = f.read()

# Fix Plotly charts to respect the template by passing theme=None to st.plotly_chart
code = re.sub(r'(st\.plotly_chart\([^,]+, use_container_width=True)', r'\1, theme=None', code)

# Let's also check if there are other calls
code = re.sub(r'(st\.plotly_chart\(.*?)(, key=)', r'\1, theme=None\2', code)

with open("app.py", "w") as f:
    f.write(code)
