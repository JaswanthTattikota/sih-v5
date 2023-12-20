from flask import Flask, request, jsonify
import json
import pandas as pd
# import datapre as dp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load your dataset (replace 'your_dataset.csv' with your actual file)
df = pd.read_csv('Coursedetails.csv')
df.info()

def preprocess(df):
# Drop duplicates if any
  df = df.drop_duplicates()

# Convert 'Course Rating' to numeric (assuming it contains ratings)
  # df['Course Rating'] = pd.to_numeric(df['Course Rating'], errors='coerce')

# Fill missing values or handle them based on your preference
# For simplicity, we'll fill missing ratings with the mean rating
  # df['Course Rating'].fillna(df['Course Rating'].mean(), inplace=True)

# Combine relevant columns into a single text column for text-based analysis
  df['Combined Features'] = df['Course Name'] + ' '  + df['Course Description'] + ' ' + df['Skills']

# Display the processed data
  print(df.head())
  return df

df = preprocess(df)

# Use TF-IDF vectorizer to convert text into numerical features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Combined Features'])

# Display the shape of the TF-IDF matrix
print(tfidf_matrix.shape)


# Calculate the cosine similarity between courses
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Display the shape of the cosine similarity matrix
print(cosine_sim.shape)

# Function to get course recommendations based on cosine similarity
def get_course_recommendations(course_name, cosine_sim_matrix, df, top_n=5):
    # Get the index of the course
    course_index = df[df['Course Name'] == course_name].index[0]

    # Get the cosine similarity scores for the course
    sim_scores = list(enumerate(cosine_sim_matrix[course_index]))

    # Sort the courses based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top-n similar courses
    top_indices = [x[0] for x in sim_scores[1:top_n+1]]

    # Return the names of the top-n similar courses
    return df['Course Name'].iloc[top_indices]

# Example: Get recommendations for a specific course
@app.route('/', methods = ['POST'])

   
# @app.route('/recommend_course', methods = ['POST'])
def recommend_courses():
  # courses = 'Programming Languages, Part A'
  # course_name = requestname()
  courses = request.get_json('course')
  course_name = courses.get('course')
  recommendations = get_course_recommendations(course_name, cosine_sim, df)
# print(f"Recommendations for '{course_name}': {recommendations.values}")
  return jsonify({'recommendations': recommendations.tolist()})


