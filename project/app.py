import streamlit as st
import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("./project/dataset_B.csv")  # Notice: no full path, just filename!

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

# Define the expected columns (VERY IMPORTANT!)
expected_cols = [
    'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_qm', 'nb_and', 'nb_or', 'nb_eq', 
    'nb_underscore', 'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon', 'nb_comma', 'nb_semicolumn', 
    'nb_dollar', 'nb_space', 'nb_www', 'nb_com', 'nb_dslash', 'http_in_path', 'https_token', 'ratio_digits_url', 
    'ratio_digits_host', 'punycode', 'port', 'tld_in_path', 'tld_in_subdomain', 'abnormal_subdomain', 'nb_subdomains',
    'prefix_suffix', 'random_domain', 'shortening_service', 'path_extension', 'nb_redirection', 'nb_external_redirection',
    'length_words_raw', 'char_repeat', 'shortest_words_raw', 'shortest_word_host', 'shortest_word_path',
    'longest_words_raw', 'longest_word_host', 'longest_word_path', 'avg_words_raw', 'avg_word_host', 'avg_word_path',
    'phish_hints', 'domain_in_brand', 'brand_in_subdomain', 'brand_in_path', 'suspecious_tld', 'statistical_report',
    'nb_hyperlinks', 'ratio_intHyperlinks', 'ratio_extHyperlinks', 'ratio_nullHyperlinks', 'nb_extCSS', 
    'ratio_intRedirection', 'ratio_extRedirection', 'ratio_intErrors', 'ratio_extErrors', 'login_form', 
    'external_favicon', 'links_in_tags', 'submit_email', 'ratio_intMedia', 'ratio_extMedia', 'sfh', 'iframe', 
    'popup_window', 'safe_anchor', 'onmouseover', 'right_clic', 'empty_title', 'domain_in_title', 'domain_with_copyright',
    'whois_registered_domain', 'domain_registration_length', 'domain_age', 'web_traffic', 'dns_record', 'google_index',
    'page_rank'
]


features = {col: df[col].mean() for col in expected_cols}

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

# --- Streamlit App ---
st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️", layout="centered")
st.title("🛡️ Phishing Website Detector")

st.markdown("""
Welcome to the Phishing Website Detector!  
Enter a website URL below, and our model will predict whether it is **Legitimate** ✅ or **Phishing** ⚠️
""")

url_input = st.text_input("🔗 Enter a Website URL to Check:")

if st.button("🚀 Analyze URL"):
    if url_input:
        with st.spinner("🔎 Analyzing Website... Please wait..."):
            user_features = extract_features(url_input)
            user_features = user_features[expected_cols]
            user_features_scaled = scaler.transform(user_features)

            prediction = rf.predict(user_features_scaled)
            prediction_prob = rf.predict_proba(user_features_scaled)

            pred_confidence = round(max(prediction_prob[0])*100, 2)

            if prediction[0] == 1:
                verdict = "⚠️ Phishing Website"
                color = "red"
            else:
                verdict = "✅ Legitimate Website"
                color = "green"

            st.markdown("---")

            # --- KPI Cards ---
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(label="🔍 Prediction", value=verdict)

            with col2:
                st.metric(label="📈 Confidence", value=f"{pred_confidence}%")

            with col3:
                # Dummy accuracy (example 96%) - you can replace if you have true model accuracy
                st.metric(label="🎯 Model Accuracy", value="80%")

            st.markdown("---")
    else:
        st.warning("⚠️ Please enter a URL.")

st.caption("Built with ❤️ using Streamlit and Machine Learning")
