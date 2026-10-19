import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Page Config
st.set_page_config(page_title="Hybrid Recommender System", layout="wide")

st.title("Fast Hybrid Recommender System")
st.write("Optimized version for lightning-fast recommendations.")

# Sidebar Navigation
option = st.sidebar.selectbox("Choose Recommender Domain", ["Movie Recommender", "Product Recommender"])

# ==================== 1. MOVIE RECOMMENDER ====================
if option == "Movie Recommender":
    st.subheader("Movie Recommender (Optimized)")

    @st.cache_resource
    def load_and_train_movies():
        if os.path.exists('movies.csv') and os.path.exists('ratings.csv'):
            movies_df = pd.read_csv('movies.csv')
            # Sirf pehli 50,000 rows load karenge taaki bilkul fast chale
            ratings_df = pd.read_csv('ratings.csv', nrows=50000)
            
            # 1. Content-based setup
            movies_df['genres'] = movies_df['genres'].fillna('')
            tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
            tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
            content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # 2. Collaborative Filtering setup (Optimized & Fast)
            n_ratings = ratings_df.groupby('movieId').size()
            popular_movies = n_ratings[n_ratings >= 5].index
            filtered_ratings = ratings_df[ratings_df['movieId'].isin(popular_movies)]
            
            pivot = filtered_ratings.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
            sparse_pivot = csr_matrix(pivot.values)
            
            model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
            model_knn.fit(sparse_pivot)
            
            return movies_df, content_sim, pivot, model_knn
        return None, None, None, None

    with st.spinner('Loading and optimizing models... Please wait.'):
        movies_df, content_sim, pivot, model_knn = load_and_train_movies()

    if movies_df is not None:
        rec_method = st.radio("Select Method:", ["Content-Based (Genres)", "Collaborative Filtering (User Ratings)"])
        selected_movie = st.selectbox("Search or Select a Movie:", movies_df['title'].values)

        if rec_method == "Content-Based (Genres)":
            if st.button('Recommend Movies'):
                idx = movies_df[movies_df['title'] == selected_movie].index[0]
                sim_scores = list(enumerate(content_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
                movie_indices = [i[0] for i in sim_scores]
                
                st.success("Top Similar Movies (By Genre):")
                for i, rec in enumerate(movies_df['title'].iloc[movie_indices].tolist(), 1):
                    st.write(f"{i}. {rec}")
                    
        else:
            if st.button('Recommend Movies'):
                try:
                    movie_id = movies_df[movies_df['title'] == selected_movie]['movieId'].values[0]
                    if movie_id in pivot.index:
                        movie_idx = pivot.index.get_loc(movie_id)
                        distances, indices = model_knn.kneighbors(pivot.iloc[movie_idx, :].values.reshape(1, -1), n_neighbors=6)
                        
                        st.success("Top Similar Movies (By User Rating Behavior):")
                        for i, idx in enumerate(indices.flatten()[1:], 1):
                            recommended_movie_id = pivot.index[idx]
                            rec_title = movies_df[movies_df['movieId'] == recommended_movie_id]['title'].values[0]
                            st.write(f"{i}. {rec_title}")
                    else:
                        st.warning("Is movie ke liye sufficient ratings available nahi hain, kripya Content-Based try karein.")
                except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.error("Error: 'movies.csv' ya 'ratings.csv' folder mein nahi mili!")

# ==================== 2. PRODUCT RECOMMENDER ====================
elif option == "Product Recommender":
    st.subheader("Product Recommender System")

    @st.cache_resource
    def load_product_data():
        data = {
            'product_name': [
                'ASUS VivoBook S14 OLED Laptop', 'MacBook Air M2', 'Dell Inspiron 15', 'Lenovo IdeaPad Slim 3',
                'Sony WH-1000XM5 Headphones', 'JBL Charge 5 Bluetooth Speaker', 'Boat Rockerz 450',
                'Nike Air Zoom Running Shoes', 'Adidas Ultraboost Light', 'Puma Casual Smash Sneakers',
                'Samsung Galaxy S24 Ultra', 'OnePlus 12 5G', 'Apple iPhone 15 Pro', 'Redmi Note 13 Pro'
            ],
            'category': [
                'Laptop Electronics Computer Windows', 'Laptop Electronics Computer Mac Apple', 'Laptop Electronics Computer Windows', 'Laptop Electronics Computer Windows',
                'Audio Headphones Music Wireless', 'Audio Speaker Music Bluetooth', 'Audio Headphones Music Wireless',
                'Footwear Sports Running Shoes', 'Footwear Sports Running Shoes', 'Footwear Casual Sneakers Shoes',
                'Mobile Electronics Phone Smartphone Android', 'Mobile Electronics Phone Smartphone Android', 'Mobile Electronics Phone Smartphone iOS', 'Mobile Electronics Phone Smartphone Android'
            ],
            'price': ['₹75,000', '₹1,10,000', '₹55,000', '₹40,000', '₹29,990', '₹14,999', '₹1,499', '₹8,995', '₹14,999', '₹2,999', '₹1,29,999', '₹64,999', '₹1,34,999', '₹25,999']
        }
        df_prod = pd.DataFrame(data)
        tfidf_p = TfidfVectorizer(stop_words='english')
        tfidf_matrix_p = tfidf_p.fit_transform(df_prod['category'])
        cosine_sim_p = cosine_similarity(tfidf_matrix_p, tfidf_matrix_p)
        return df_prod, cosine_sim_p

    df_products, cosine_sim_products = load_product_data()

    selected_product = st.selectbox("Search or Select a Product:", df_products['product_name'].values)

    if st.button('Recommend Products'):
        idx = df_products[df_products['product_name'] == selected_product].index[0]
        sim_scores = list(enumerate(cosine_sim_products[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]
        
        st.success("Recommended Products for You:")
        for i, score in enumerate(sim_scores, 1):
            prod_idx = score[0]
            name = df_products.loc[prod_idx, 'product_name']
            cat = df_products.loc[prod_idx, 'category']
            price = df_products.loc[prod_idx, 'price']
            st.info(f"**{i}. {name}** \n* Category: {cat} \n* Price: {price}")

## python -m streamlit run app.py