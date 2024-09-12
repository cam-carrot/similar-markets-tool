from flask import Flask, render_template, request, flash, jsonify
import folium
from engine import MarketAnalysisEngine, MARKET_TAGS  # Import MARKET_TAGS from engine.py
import logging
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['DEBUG'] = True

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

engine = MarketAnalysisEngine("cities.csv", "ga4data.csv")

# Custom filter for number formatting
@app.template_filter('format_number')
def format_number(value):
    return "{:,}".format(int(value))

def create_map(similar_cities, target_city, target_state):
    target_city_state = f"{target_city}, {target_state}".lower().strip()
    
    target_lat, target_lon = similar_cities.loc[target_city_state, ['lat', 'lng']]
    m = folium.Map(location=[target_lat, target_lon], zoom_start=8)

    for idx, row in similar_cities.iterrows():
        color = 'red' if idx == target_city_state else \
                'green' if row['opportunity_category'] == 'High' else \
                'orange' if row['opportunity_category'] == 'Average' else 'blue'
        
        folium.Marker(
            [row['lat'], row['lng']],
            popup=f"{idx}<br>Opportunity: {row['opportunity_category']}",
            tooltip=idx,
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)

    return m._repr_html_()

@app.route('/')
def index():
    logger.info("Index route accessed")
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        target_city = request.form['city']
        target_state = request.form['state']
        radius = int(request.form['radius'])  # Get the radius from the form
        
        app.logger.info(f"Analyzing market for {target_city}, {target_state} with radius {radius} miles")
        
        similar_cities = engine.find_similar_cities(target_city, target_state, radius_miles=radius)
        app.logger.info(f"Found {len(similar_cities)} similar cities")
        
        # Convert DataFrame to list of dictionaries
        similar_cities_list = similar_cities.to_dict('records')
        app.logger.debug(f"Similar cities data: {similar_cities_list}")
        
        target_data = similar_cities.loc[f"{target_city}, {target_state}".lower()].to_dict()
        app.logger.debug(f"Target data: {target_data}")
        
        map_html = create_map(similar_cities, target_city, target_state)
        
        return render_template('results.html',
                               target_city=target_city,
                               target_state=target_state,
                               target_data=target_data,
                               similar_cities=similar_cities_list,
                               map_html=map_html,
                               market_tags=MARKET_TAGS)  # Pass MARKET_TAGS to the template
    except ValueError as e:
        if "not found in the dataset" in str(e):
            app.logger.warning(f"City not found: {target_city}, {target_state}")
            return render_template('cityerror.html', city=target_city, state=target_state)
        else:
            app.logger.error(f"Error in analyze route: {str(e)}", exc_info=True)
            return render_template('cityerror.html', error_message=str(e))
    except Exception as e:
        app.logger.error(f"Error in analyze route: {str(e)}", exc_info=True)
        return render_template('cityerror.html', error_message=str(e))

@app.errorhandler(404)
def page_not_found(e):
    logger.error(f"404 error: {request.url}")
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)