import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import logging

class MarketAnalysisEngine:
    def __init__(self, city_data_path, ga4_data_path):
        # Setting up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("Initializing MarketAnalysisEngine")
        
        # Loading and standardizing city data
        self.city_data = pd.read_csv(city_data_path, low_memory=False)
        self.standardize_city_names(self.city_data)
        self.logger.info(f"Loaded city data. Shape: {self.city_data.shape}")
        
        # Loading and standardizing GA4 data
        self.ga4_data = pd.read_csv(ga4_data_path, low_memory=False)
        self.standardize_city_names(self.ga4_data)
        self.logger.info(f"Loaded GA4 data. Shape: {self.ga4_data.shape}")
        
        self.prepare_data()

    def standardize_city_names(self, df):
        if 'city' in df.columns and 'state_id' in df.columns:
            df['city_state'] = df['city'] + ', ' + df['state_id']
        elif 'city_state' in df.columns:
            pass  # city_state column already exists
        else:
            self.logger.error("Required columns for city_state not found in dataframe")
            raise ValueError("DataFrame must have 'city' and 'state_id' columns or a 'city_state' column")
        
        df['city_state'] = df['city_state'].str.lower().str.strip()
        self.logger.info(f"Standardized city names. Sample: {df['city_state'].head().tolist()}")

    def prepare_data(self):
        # Set the standardized city_state as index for city_data
        self.city_data.set_index('city_state', inplace=True)
        self.logger.info(f"Total cities in dataset: {len(self.city_data)}")
        
        # Ensure GA4 data has a standardized city_state column
        if 'city_state' not in self.ga4_data.columns:
            self.logger.error("GA4 data does not have a city_state column")
            raise ValueError("GA4 data must have a city_state column")

    def find_similar_cities(self, target_city, target_state, radius_miles=100, n_similar=15, feature_weights=None):
        target_city_state = f"{target_city}, {target_state}".lower().strip()
        self.logger.info(f"Finding similar cities for {target_city_state}")

        if target_city_state not in self.city_data.index:
            self.logger.error(f"Target city '{target_city_state}' not found in the dataset")
            raise ValueError(f"Target city '{target_city_state}' not found in the dataset")

        nearby_cities = self.filter_cities_by_distance(target_city_state, radius_miles)
        self.logger.info(f"Cities within {radius_miles} miles: {len(nearby_cities)}")

        features = [
            'population', 'population_proper', 'density', 'incorporated', 'age_median',
            'age_over_65', 'family_dual_income', 'income_household_median', 'income_household_six_figure',
            'home_ownership', 'housing_units', 'home_value', 'rent_median', 'education_college_or_above',
            'race_white', 'race_black', 'hispanic', 'income_individual_median', 'rent_burden', 'poverty'
        ]

        nearby_cities = self.clean_data(nearby_cities, features)
        self.logger.info(f"Shape of nearby_cities after cleaning: {nearby_cities.shape}")

        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(nearby_cities[features])

        # Use the provided feature weights or default to equal weights
        if feature_weights is None:
            feature_weights = {feature: 1 for feature in features}
        
        # Ensure all features have a weight (use 1 as default if not specified)
        weights = np.array([feature_weights.get(feature, 1) for feature in features])
        
        self.logger.info(f"Using feature weights: {feature_weights}")
        
        weighted_data = normalized_data * weights.reshape(1, -1)  # Reshape weights to match normalized_data shape
        
        self.logger.info(f"Shape of weighted data: {weighted_data.shape}")

        nn = NearestNeighbors(n_neighbors=min(n_similar, len(nearby_cities)), metric='euclidean')
        nn.fit(weighted_data)

        target_index = nearby_cities.index.get_loc(target_city_state)
        distances, indices = nn.kneighbors(weighted_data[target_index].reshape(1, -1))

        similar_cities = nearby_cities.iloc[indices[0]].copy()

        self.logger.info(f"Similar cities before ensuring target city: {similar_cities.index.tolist()}")

        similar_cities['distance_to_target'] = self.haversine_distances(
            similar_cities[['lat', 'lng']].values,
            self.city_data.loc[target_city_state, ['lat', 'lng']].values.flatten()
        )
        similar_cities['similarity_score'] = np.where(similar_cities.index == target_city_state, 0, distances[0])

        ga4_columns = ['users_org', 'cvr_org', 'leads_cvr', 'users_paid', 'cvr_paid', 'leads_paid']
        self.logger.info(f"GA4 data shape before merge: {self.ga4_data.shape}")
        self.logger.info(f"Similar cities shape before merge: {similar_cities.shape}")
        
        self.logger.info(f"Sample of GA4 data:\n{self.ga4_data.head().to_string()}")
        self.logger.info(f"Sample of similar cities before merge:\n{similar_cities.head().to_string()}")
        
        similar_cities = similar_cities.merge(self.ga4_data[['city_state'] + ga4_columns], left_index=True, right_on='city_state', how='left')
        self.logger.info(f"Similar cities shape after merge: {similar_cities.shape}")
        
        for col in ga4_columns:
            nan_count = similar_cities[col].isna().sum()
            self.logger.info(f"NaN count in {col} after merge: {nan_count}")
        
        similar_cities.set_index('city_state', inplace=True)
        self.logger.info(f"Columns in similar_cities after merge: {similar_cities.columns.tolist()}")
        
        self.logger.info(f"Sample of similar cities after merge:\n{similar_cities.head().to_string()}")

        similar_cities, std_ga4_columns = self.calculate_opportunity_score(similar_cities, target_city_state)
        similar_cities = similar_cities.sort_values('opportunity_score', ascending=False)

        self.logger.info(f"Final similar cities: {similar_cities.index.tolist()}")
        self.logger.info(f"Is target city in results: {target_city_state in similar_cities.index}")
        return similar_cities

    def clean_data(self, df, features):
        for feature in features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
        imputer = SimpleImputer(strategy='median')
        df[features] = imputer.fit_transform(df[features])
        return df

    def filter_cities_by_distance(self, target_city_state, radius_miles):
        target_lat, target_lon = self.city_data.loc[target_city_state, ['lat', 'lng']]
        distances = self.haversine_distances(self.city_data[['lat', 'lng']].values, np.array([target_lat, target_lon]))
        nearby_mask = distances <= radius_miles
        return self.city_data[nearby_mask].copy()

    @staticmethod
    def haversine_distances(points, target):
        R = 3959.87433  # Earth's radius in miles
        if points.ndim == 1:
            points = points.reshape(1, -1)
        lat1, lon1 = np.radians(points[:, 0]), np.radians(points[:, 1])
        lat2, lon2 = np.radians(target[0]), np.radians(target[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def calculate_opportunity_score(self, df, target_city_state):
        self.logger.info(f"Calculating opportunity score. DataFrame shape: {df.shape}")
        
        if df.empty:
            self.logger.error("DataFrame is empty in calculate_opportunity_score")
            raise ValueError("No data available to calculate opportunity score")

        ga4_columns = ['users_org', 'cvr_org', 'leads_cvr', 'users_paid', 'cvr_paid', 'leads_paid']
        
        missing_columns = [col for col in ga4_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(f"Missing GA4 columns: {missing_columns}")
            raise ValueError(f"Missing GA4 columns: {missing_columns}")

        for col in ga4_columns:
            nan_count = df[col].isna().sum()
            self.logger.info(f"NaN count in {col}: {nan_count}")

        scaler = StandardScaler()
        std_ga4_columns = [f'std_{col}' for col in ga4_columns]
        df[std_ga4_columns] = scaler.fit_transform(df[ga4_columns].fillna(df[ga4_columns].mean()))

        avg_performance = df[df.index != target_city_state][std_ga4_columns].mean()
        df['performance_diff'] = (df[std_ga4_columns] - avg_performance).mean(axis=1)
        
        if 'similarity_score' not in df.columns:
            self.logger.error("'similarity_score' column not found in DataFrame")
            raise ValueError("'similarity_score' column not found in DataFrame")
        
        if df['similarity_score'].isna().all():
            self.logger.warning("All similarity scores are NaN. Setting norm_similarity to 1.")
            df['norm_similarity'] = 1
        else:
            df['norm_similarity'] = 1 - (df['similarity_score'] - df['similarity_score'].min()) / (df['similarity_score'].max() - df['similarity_score'].min())

        df['raw_opportunity_score'] = df['norm_similarity'] * (1 - df['performance_diff'])
        
        if 'housing_units' not in df.columns:
            self.logger.error("'housing_units' column not found in DataFrame")
            raise ValueError("'housing_units' column not found in DataFrame")
        
        zero_housing = (df['housing_units'] == 0).sum()
        if zero_housing > 0:
            self.logger.warning(f"{zero_housing} cities have zero housing units. Replacing with 1 to avoid division by zero.")
            df['housing_units'] = df['housing_units'].replace(0, 1)
        
        df['opportunity_score'] = df['raw_opportunity_score'] / df['housing_units']
        
        if df['opportunity_score'].isna().all():
            self.logger.warning("All opportunity scores are NaN. Setting to a constant value.")
            df['opportunity_score'] = 1
        else:
            min_score = df['opportunity_score'].min()
            max_score = df['opportunity_score'].max()
            if min_score == max_score:
                self.logger.warning("All non-NaN opportunity scores are the same. Setting to a constant value.")
                df['opportunity_score'] = 1
            else:
                df['opportunity_score'] = (df['opportunity_score'] - min_score) / (max_score - min_score)

        self.logger.info(f"Opportunity score statistics: min={df['opportunity_score'].min()}, max={df['opportunity_score'].max()}, mean={df['opportunity_score'].mean()}")

        unique_scores = df['opportunity_score'].nunique()
        if unique_scores < 3:
            self.logger.warning(f"Not enough unique opportunity scores ({unique_scores}) for qcut. Using manual categorization.")
            df['opportunity_category'] = pd.cut(df['opportunity_score'], bins=[-float('inf'), 0.33, 0.67, float('inf')], labels=['Low', 'Average', 'High'])
        else:
            df['opportunity_category'] = pd.qcut(df['opportunity_score'], q=3, labels=['Low', 'Average', 'High'])

        self.logger.info("Opportunity score calculation completed successfully")
        return df, std_ga4_columns