from flask import render_template, request
from app import app
from app.project import combined_prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        news_text = request.form.get('news')
        if news_text:
            result = combined_prediction(news_text)
    return render_template('index.html', result=result)
