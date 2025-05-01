
# from flask import Flask, jsonify, render_template
# from run import main
# import os

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/run', methods=['POST'])
# def run_optimization():
#     pop, _, _ = main()
#     returns = [ind.fitness.values[0] for ind in pop]
#     volatility = [-ind.fitness.values[1] for ind in pop]
#     esg = [ind.fitness.values[2] for ind in pop]
#     return jsonify({'returns': returns, 'volatility': volatility, 'esg': esg})

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port, debug=True)

from flask import Flask, jsonify, render_template, send_file
from run_moead import main, calculate_spacing
import numpy as np
import os
import pandas as pd
from io import BytesIO
from pymoo.indicators.hv import HV

app = Flask(__name__)

last_pop = None
last_stocks = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/methods')
def methods():
    return render_template('methods.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/MOEAD')
def MOEAD():
    return render_template('moead.html')

@app.route('/PSO')
def PSO():
    return render_template('PSO.html')

@app.route('/SPEA')
def SPEA():
    return render_template('SPEA.html')

@app.route('/NSGA2')
def NSGA2():
    return render_template('NSGA2.html')

@app.route('/result_SPEA')
def result_SPEA():
    return render_template('result_SPEA.html')



@app.route('/run', methods=['POST'])
def run_optimization():
    global last_pop, last_stocks
    pop, _, hof, valid_stocks = main()
    last_pop = pop
    last_stocks = valid_stocks
    fitness_values = np.array([ind.fitness.values for ind in pop])
    ref_point = [0.01,0.002,50]
    ind = HV(ref_point=ref_point)
    hv = ind(fitness_values)
    spacing = calculate_spacing(fitness_values)
    returns = [ind.fitness.values[0] for ind in pop]
    volatility = [-ind.fitness.values[1] for ind in pop]
    esg = [ind.fitness.values[2] for ind in pop]

    return jsonify({'returns': returns, 'volatility': volatility, 'esg': esg,'hypervolume': round(hv, 4),'spacing': round(spacing, 4)})

@app.route('/download_weights', methods=['GET'])
def download_weights():
    if last_pop is None or last_stocks is None:
        return "You must run optimization first!", 400

    data = []
    for ind in last_pop:
        weights = np.array(ind)
        weights /= np.sum(weights)
        data.append(weights)

    df = pd.DataFrame(data, columns=last_stocks)

    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='portfolio_weights.csv')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)