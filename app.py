from flask import Flask, render_template, request, flash, jsonify
import folium
from engine import MarketAnalysisEngine
import logging

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

    return m

@app.route('/')
def index():
    logger.info("Index route accessed")
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    logger.info("Analyze route accessed")
    logger.info(f"Form data: {request.form}")

    try:
        target_city = request.form['city'].lower().strip()
        target_state = request.form['state'].lower().strip()
        radius_miles = int(request.form['radius'])

        logger.info(f"Analyzing: {target_city}, {target_state} with radius {radius_miles}")

        # Define feature weights
        feature_weights = {
            'population': 5,
            'population_proper': 5,
            'housing_units': 5,
            'income_household_median': 2,
            'home_value': 2.5,
            'education_college_or_above': 2,
            'age_median': 1.5
            # Add more weights as needed
        }

        # Pass feature_weights to find_similar_cities
        similar_cities = engine.find_similar_cities(target_city, target_state, radius_miles, feature_weights=feature_weights)
        logger.info(f"Similar cities found: {len(similar_cities)}")

        target_city_state = f"{target_city}, {target_state}"
        if target_city_state not in similar_cities.index:
            logger.error(f"Target city {target_city_state} not found in similar cities after analysis")
            raise ValueError(f"Target city {target_city_state} not found in similar cities after analysis")

        target_data = similar_cities.loc[target_city_state]
        logger.info(f"Target data: {target_data.to_dict()}")

        map_html = create_map(similar_cities, target_city, target_state)._repr_html_()
        logger.info("Map created successfully")

        return render_template('results.html', 
                               map_html=map_html, 
                               target_city=target_city.title(),
                               target_state=target_state.upper(),
                               target_data=target_data,
                               similar_cities=similar_cities)
    except Exception as e:
        logger.error(f"Error in analyze route: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    logger.error(f"404 error: {request.url}")
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)