import io

import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State
import torch

from model.final import load_lstm_model, preprocess_text, vectorize_text

# Define the preprocess_data function here
def preprocess_data(data):
    # Add your preprocessing steps here
    return data

# Rest of the script
data = pd.read_excel('D:\pubg sentiment analysis\Multilingual PUBG Reviews for Sentiment Analysis.xlsx')
data = preprocess_data(data)

# Vectorize the text data
vectorizer, X_train_vectorized, X_test_vectorized = vectorize_text(data['Review_Content'], data['Review_Content'])

# Load the vectorizer
loaded_vectorizer = vectorizer

# Load the LSTM model
loaded_lstm_model = load_lstm_model(lstm_model_path)

# Define a function to predict the sentiment using the loaded LSTM model
def predict_sentiment(input_text):
    # Preprocess the input text
    input_text = preprocess_text(input_text)

    # Vectorize the preprocessed input text using the loaded vectorizer
    input_vector = loaded_vectorizer.transform([input_text])

    # Predict the sentiment using the loaded LSTM model
    with torch.no_grad():
        model_input = torch.tensor(input_vector.toarray(), dtype=torch.float32).to(torch.device)
        model_input = model_input.unsqueeze(1)  # Add a dimension for sequence length (sequence length = 1 in your case)
        output = loaded_lstm_model(model_input)
        prediction = (output.squeeze().cpu().numpy() >= 0.5).astype(int)

    # Map the prediction to the corresponding sentiment label
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment_label = sentiment_labels[prediction[0]]

    return sentiment_label

# Initialize the app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    # App title
    html.H1('PUBG Reviews Sentiment Analysis Dashboard', style={'textAlign': 'center'}),

    # Input text box for user input
    html.Div([
        html.H6('Enter a review to predict its sentiment:'),
        dcc.Input(id='user-input', value='', type='text', style={'marginTop': '10px', 'width': '50%'})
    ]),

    # Prediction button
    html.Button('Predict Sentiment', id='predict-button', style={'marginTop': '10px'}),

    # Prediction result
    html.Div(id='prediction-result', children=[]),

    # Histogram of helpful ratings
    html.Div([
        html.H6('Distribution of Helpful Ratings'),
        dcc.Graph(id='histogram-helpful-ratings', figure={})
    ]),

    # Histogram of funny ratings
    html.Div([
        html.H6('Distribution of Funny Ratings'),
        dcc.Graph(id='histogram-funny-ratings', figure={})
    ]),

    # Histogram of more ratings
    html.Div([
        html.H6('Distribution of More Ratings'),
        dcc.Graph(id='histogram-more-ratings', figure={})
    ]),

    # Scatter plot of helpful vs funny ratings
    html.Div([
        html.H6('Scatter plot: Helpful Rating vs Funny Rating'),
        dcc.Graph(id='scatter-plot-ratings',figure={})
    ]),

    # Correlation matrix
    html.Div([
        html.H6('Correlation Matrix'),
        dcc.Graph(id='correlation-matrix', figure={})
    ]),

    # Review count by recommendation tag
    html.Div([
        html.H6('Review Count by Recommendation Tag'),dcc.Graph(id='review-count-recommendation', figure={})
    ]),

    # Review count over time
    html.Div([
        html.H6('Review Count Over Time'),
        dcc.Graph(id='review-count-time', figure={})
    ]),

    # Language distribution
    html.Div([
        html.H6('Language Distribution'),
        dcc.Graph(id='language-distribution', figure={})
    ])
])

# Define the callback for the prediction button
@app.callback(
    Output('prediction-result', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('user-input', 'value')]
)
def update_prediction(n_clicks, user_input):
    if user_input:
        sentiment = predict_sentiment(user_input)
        return html.Div([
            html.H6(f'Predicted Sentiment: {sentiment}')
        ])
    else:
        return html.Div([
            html.H6('Please enter a review to predict its sentiment.')
        ])

# Define the callbacks for the plots
@app.callback(
    Output('histogram-helpful-ratings', 'figure'),
    [Input('histogram-helpful-ratings', 'relayoutData')]
)
def update_histogram_helpful_ratings(relayout_data):
    if relayout_data:
        bins = int(relayout_data['xaxis.range[0]'])
        fig = px.histogram(data, x='Helpful_Rating', nbins=bins, title='Distribution of Helpful Ratings')
        return fig
    else:
        fig = px.histogram(data, x='Helpful_Rating', title='Distribution of Helpful Ratings')
        return fig

@app.callback(
    Output('histogram-funny-ratings', 'figure'),
    [Input('histogram-funny-ratings', 'relayoutData')]
)
def update_histogram_funny_ratings(relayout_data):
    if relayout_data:
        bins = int(relayout_data['xaxis.range[0]'])
        fig = px.histogram(data, x='Funny_Rating', nbins=bins, title='Distribution of Funny Ratings')
        return fig
    else:
        fig = px.histogram(data, x='Funny_Rating', title='Distribution of Funny Ratings')
        return fig

@app.callback(
    Output('histogram-more-ratings', 'figure'),
    [Input('histogram-more-ratings', 'relayoutData')]
)
def update_histogram_more_ratings(relayout_data):
    if relayout_data:
        bins = int(relayout_data['xaxis.range[0]'])
        fig = px.histogram(data, x='More_Rating', nbins=bins, title='Distribution of More Ratings')
        return fig
    else:
        fig = px.histogram(data, x='More_Rating', title='Distribution of More Ratings')
        return fig

@app.callback(
    Output('scatter-plot-ratings', 'figure'),
    [Input('scatter-plot-ratings', 'relayoutData')]
)
def update_scatter_plot_ratings(relayout_data):
    if relayout_data:
        x_range = relayout_data['xaxis.range']
        y_range = relayout_data['yaxis.range']
        fig = px.scatter(data, x='Helpful_Rating', y='Funny_Rating', title='Scatter plot: Helpful Rating vs Funny Rating')
        fig.update_xaxes(range=x_range)
        fig.update_yaxes(range=y_range)
        return fig
    else:
        fig = px.scatter(data, x='Helpful_Rating', y='Funny_Rating', title='Scatter plot: Helpful Rating vs Funny Rating')
        return fig

@app.callback(
    Output('correlation-matrix', 'figure'),
    [Input('correlation-matrix', 'relayoutData')]
)
def update_correlation_matrix(relayout_data):
    if relayout_data:
        fig = px.histogram(data[['Helpful_Rating', 'Funny_Rating', 'More_Rating']].corr(), title='Correlation Matrix')
        return fig
    else:
        fig = px.histogram(data[['Helpful_Rating', 'Funny_Rating', 'More_Rating']].corr(), title='Correlation Matrix')
        return fig

@app.callback(
    Output('review-count-recommendation', 'figure'),
    [Input('review-count-recommendation', 'relayoutData')]
)
def update_review_count_recommendation(relayout_data):
    if relayout_data:
        fig = px.bar(data, x='Recommend_Tag', y=data.index, title='Review Count by Recommendation Tag')
        return fig
    else:
        fig = px.bar(data, x='Recommend_Tag', y=data.index, title='Review Count by Recommendation Tag')
        return fig

@app.callback(
    Output('review-count-time', 'figure'),
    [Input('review-count-time', 'relayoutData')]
)
def update_review_count_time(relayout_data):
    if relayout_data:
        fig = px.line(data, x='Year-Month', y='Review_Count', title='Review Count Over Time')
        return fig
    else:
        fig = px.line(data, x='Year-Month', y='Review_Count', title='Review Count Over Time')
        return fig

@app.callback(
    Output('language-distribution', 'figure'),
    [Input('language-distribution', 'relayoutData')]
)
def update_language_distribution(relayout_data):
    if relayout_data:
        fig = px.pie(data['Language_Tag'].value_counts(), title='Language Distribution')
        return fig
    else:
fig = px.pie(data['Language_Tag'].value_counts(), title='Language Distribution')
        return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)