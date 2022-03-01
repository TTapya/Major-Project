from flask import Flask, render_template, request, url_for, redirect
import pickle
from scipy.sparse import hstack
app=Flask(__name__)


with open(f'models/final_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(f'models/final_essay_vec.pkl', 'rb') as f:
    essay_vec = pickle.load(f)

with open(f'models/final_state_vec.pkl', 'rb') as f:
    state_vec = pickle.load(f)

with open(f'models/final_grade_vec.pkl', 'rb') as f:
    grade_vec = pickle.load(f)    

with open(f'models/final_prefix_vec.pkl', 'rb') as f:
    prefix_vec = pickle.load(f)

with open(f'models/final_cat_vec.pkl', 'rb') as f:
    cat_vec = pickle.load(f)

with open(f'models/final_subcat_vec.pkl', 'rb') as f:
    subcat_vec = pickle.load(f)    

with open(f'models/final_price_norm.pkl', 'rb') as f:
    price_norm = pickle.load(f)

@app.route('/', methods = ['GET', 'POST'])
def home_page():
    if request.method == 'POST':
        return redirect(url_for('predict'))
    return render_template('home.html')

@app.route('/result', methods = ['GET', 'POST'])
def predict():
    price = float(request.form['price'])
    state = request.form['state']
    essay = (request.form['essay']).lower()
    cat = request.form['cat']
    subcat = request.form['subcat']
    grade = request.form['grade']
    prefix = request.form['prefix']

    essay_ready = essay_vec.transform([essay])
    state_ready = state_vec.transform([state])
    prefix_ready = prefix_vec.transform([prefix])
    grade_ready = grade_vec.transform([grade])
    cat_ready = cat_vec.transform([cat])
    subcat_ready = subcat_vec.transform([subcat])

    sample = hstack((essay_ready, state_ready, prefix_ready, grade_ready, cat_ready, subcat_ready, 1)).tocsr()
    prediction = round(model.predict_proba(sample)[:,1][0], 3)

    return render_template('result.html', prediction=prediction)

if __name__=="__main__":
    app.run(debug=True)