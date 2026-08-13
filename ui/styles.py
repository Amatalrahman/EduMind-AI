import streamlit as st

def inject_custom_css():
    st.markdown("""
        <style>
        /* Base Theme Colors */
        :root {
            --primary-color: #FF69B4; /* Hot Pink */
            --background-color: #FFF0F5; /* Lavender Blush (soft pink) */
            --secondary-background: #FFFFFF;
            --text-color: #333333;
        }

        /* Set App Background */
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #FFE4E1 !important; /* Misty Rose */
            border-right: 1px solid #FFC0CB;
        }

        /* Card Container Styling */
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
            background-color: var(--secondary-background);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 8px 16px rgba(255, 105, 180, 0.15);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            border: 2px solid transparent;
        }
        
        /* Interactive hover effect for cards */
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(255, 105, 180, 0.25);
            border: 2px solid #FFB6C1;
        }

        /* Standard Buttons */
        .stButton > button {
            border-radius: 12px !important;
            border: none !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            transition: all 0.2s ease !important;
        }
        
        .stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 12px rgba(255, 105, 180, 0.3) !important;
        }
        
        /* Primary Buttons */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #FF69B4, #FF1493) !important;
            color: white !important;
        }

        /* Card Images (make them fit nicely) */
        [data-testid="stImage"] img {
            border-radius: 15px;
            object-fit: cover;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #C71585 !important; /* Medium Violet Red */
            font-family: 'Inter', sans-serif;
            font-weight: 700;
        }
        </style>
    """, unsafe_allow_html=True)
