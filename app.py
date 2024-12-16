import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Engagement Prediction App", page_icon="📈")

#------------------ Data Loading and Preprocessing ------------------#
df = pd.read_csv('data/instagram_posts.csv')
df2 = pd.read_csv('data/instagram_reels.csv')

# Drop irrelevant columns
drop_cols = ['location', 'comments_disabled', 'title', 'shortcode', 'image_id', 'src', 
             'username', 'profile_image', 'followers', 'follows', 'is_private', 
             'user_id', 'content_id']
for c in drop_cols:
    if c in df.columns:
        df.drop(c, axis=1, inplace=True)

# Merge posts & reels
common_cols = [col for col in df.columns if col in df2.columns]
df_posts = df[list(common_cols)]
df_reels = df2[list(common_cols)]

df = pd.concat([df_posts, df_reels], ignore_index=True)
df.dropna(inplace=True)

df['taken_at'] = pd.to_datetime(df['taken_at'], unit='s')
df['hour'] = df['taken_at'].dt.hour
df['day_of_week'] = df['taken_at'].dt.dayofweek

df['year_week'] = df['taken_at'].dt.strftime('%Y-%U')
posts_per_week = df.groupby('year_week').size()
df['posts_per_week'] = df['year_week'].map(posts_per_week)
df.drop(columns=['year_week'], inplace=True)

def count_hashtags(caption):
    if pd.isnull(caption):
        return 0
    tags = re.findall(r'#\w+', caption)
    return len(tags)

df['num_hashtags'] = df['captions'].apply(count_hashtags)
df['word_count'] = df['captions'].apply(lambda x: len(x.split()))

X = df[['hour', 'posts_per_week', 'day_of_week', 'is_verified', 'num_hashtags', 'word_count']]
y = np.log1p(df['likes'] + 2 * df['comments_count'])

# Remove unused columns
to_drop = ['captions', 'name', 'likes', 'comments_count']
for col in to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True, errors='ignore')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
categorical_features = ['hour', 'day_of_week', 'is_verified']
numerical_features = ['num_hashtags', 'word_count', 'posts_per_week']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

rf_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=250, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
    
#------------------ Streamlit UI ------------------#
st.title("Instagram Engagement Prediction 📱🎨")

st.markdown("""
**Welcome!**  
This tool predicts the engagement ratio of a new Instagram post or reel based on selected features.  
After making a prediction, we'll show you how the predicted engagement compares to historical performance.
""")

# User inputs
st.sidebar.header("Adjust Input Features:")
hour = st.sidebar.selectbox("Hour of the day (0-23):", options=list(range(24)), index=22)
day_of_week = st.sidebar.selectbox("Day of the week (0=Monday):", options=list(range(7)), index=2)
is_verified = st.sidebar.selectbox("Is the account verified?", options=[False, True], index=1)
num_hashtags = st.sidebar.slider("Number of hashtags in caption:", 0, 32, 5)
word_count = st.sidebar.slider("Word count of caption:", 0, 300, 88)
posts_per_week = st.sidebar.slider("Number of posts this week:", 0, 100, 17)

if st.sidebar.button("Predict Engagement"):
    # Prepare input data
    input_data = pd.DataFrame([{
        'hour': hour,
        'posts_per_week': posts_per_week,
        'day_of_week': day_of_week,
        'is_verified': is_verified,
        'num_hashtags': num_hashtags,
        'word_count': word_count
    }])
    
    prediction = rf_pipeline.predict(input_data)[0]
    weighted_engagement = np.expm1(prediction)

    # Since likes and comments can't be fractional, we provide realistic scenarios:
    # Scenario 1: If all engagement came from likes (no comments):
    # likes ≈ round(weighted_engagement)
    approx_likes = int(round(weighted_engagement))

    # Scenario 2: If all engagement came from comments (no likes):
    # likes + 2 * comments = weighted_engagement
    # -> comments = weighted_engagement / 2
    approx_comments = int(round(weighted_engagement / 2))

    st.write(f"**Predicted Engagement Ratio:** {prediction:.4f}")
    st.write("""
    This ratio is based on the formula:  
    engagement_ratio = log1p(likes + 2 × comments)
    """)

    st.write(f"Reversing the formula to interpret the result:")
    st.latex(rf"likes + 2 \times comments = e^{{{prediction:.4f}}} - 1 \approx {weighted_engagement:.4f}")

    st.write(f"""
    **What does this mean in practical terms?**

    - If all your engagement were from likes only, you'd need about **{approx_likes} likes** to reach this engagement level.
    - If all your engagement were from comments only, you'd need about **{approx_comments} comments** to reach this engagement level (since each comment counts as two likes in this formula).

    These scenarios give you a more intuitive, integer-based understanding of the predicted engagement.
    """)

    historical_engagement = y.values
    historical_index = np.arange(len(historical_engagement))

    fig = go.Figure()

    # Add historical engagement line
    fig.add_trace(go.Scatter(
        x=historical_index, 
        y=historical_engagement,
        mode='lines',
        name='Historical Engagement'
    ))

    # Add prediction point with star symbol
    fig.add_trace(go.Scatter(
        x=[len(historical_engagement)],
        y=[prediction],
        mode='markers',
        name='Predicted Engagement',
        marker=dict(size=12, color='red', symbol='star')
    ))

    # Add annotation pointing to predicted value
    fig.add_annotation(
        x=len(historical_engagement), 
        y=prediction,
        text="Predicted Value",
        showarrow=True,
        arrowhead=2,
        arrowcolor='red',
        ax=0,
        ay=-40
    )

    fig.update_layout(
        title="Historical Engagement vs. Predicted Value",
        xaxis_title="Post Index",
        yaxis_title="Engagement Ratio",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**Interpretation:** If your predicted engagement is higher than most historical values, congratulations on reaching a new height! If it's lower, it might be time to reconsider your strategy.")
