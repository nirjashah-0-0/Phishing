import streamlit as st
import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("dataset_B.csv")  # Notice: no full path, just filename!

# Preprocessing
df_model = df.drop(columns=["url"])
le = LabelEncoder()
df_model["status"] = le.fit_transform(df_model["status"])

X = df_model.drop("status", axis=1)
y = df_model["status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature Extraction Function
def extract_features(url):
    parsed = urlparse(url)
    features = {col: df[col].mean() for col in expected_cols}
    features['length_url'] = len(url)
    features['length_hostname'] = len(parsed.netloc)
    features['ip'] = 1 if re.match(r'(\d{1,3}\.){3}\d{1,3}', parsed.netloc) else 0
    features['nb_dots'] = url.count('.')
    features['nb_hyphens'] = url.count('-')
    features['nb_at'] = url.count('@')
    features['nb_qm'] = url.count('?')
    features['nb_and'] = url.count('&')
    features['nb_or'] = url.count('|')
    features['nb_eq'] = url.count('=')
    features['nb_underscore'] = url.count('_')
    features['nb_tilde'] = url.count('~')
    features['nb_percent'] = url.count('%')
    features['nb_slash'] = url.count('/')
    features['nb_star'] = url.count('*')
    features['nb_colon'] = url.count(':')
    features['nb_comma'] = url.count(',')
    features['nb_semicolumn'] = url.count(';')
    features['nb_dollar'] = url.count('$')
    features['nb_space'] = url.count(' ')
    features['nb_www'] = url.count('www')
    features['nb_com'] = url.count('.com')
    features['nb_dslash'] = url.count('//')
    features['http_in_path'] = 1 if 'http' in parsed.path else 0
    features['https_token'] = 1 if 'https' in url.lower().replace("https://", "") else 0
    features['ratio_digits_url'] = sum(c.isdigit() for c in url) / len(url)
    features['ratio_digits_host'] = sum(c.isdigit() for c in parsed.netloc) / len(parsed.netloc) if parsed.netloc else 0
    features['punycode'] = 1 if 'xn--' in url else 0
    features['port'] = 1 if ':' in parsed.netloc else 0
    features['nb_subdomains'] = parsed.netloc.count('.') - 1 if parsed.netloc else 0

    return pd.DataFrame([features])

# --- Streamlit App Interface ---

st.title("🔎 Phishing Website Detector")

url_input = st.text_input("Enter the URL you want to check:")

if st.button("Check URL"):
    if url_input:
        user_features = extract_features(url_input)
        user_features = user_features[expected_cols]
        user_features_scaled = scaler.transform(user_features)

        prediction = rf.predict(user_features_scaled)
        prediction_prob = rf.predict_proba(user_features_scaled)

        if prediction[0] == 1:
            st.error(f"⚠️ This website is likely **Phishing**! ({round(prediction_prob[0][1]*100,2)}% confidence)")
        else:
            st.success(f"✅ This website appears **Legitimate**! ({round(prediction_prob[0][0]*100,2)}% confidence)")
    else:
        st.warning("⚠️ Please enter a URL.")
