import streamlit as st
import pandas as pd
from predict import predict_new_data

st.set_page_config(page_title="Marketing Conversion Predictor", page_icon="🎯", layout="wide")

st.title("🎯 Social Media Conversion Predictor")
st.write('Add info for new client to show if conversion?')

st.divider()

st.subheader("📋 Enter Customer & Campaign Details")

col1, col2, col3 = st.columns(3)

with col1:
    customer_id = st.text_input("Customer ID", value="CUST-1001")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    income = st.number_input("Income ($)", min_value=0, max_value=200000, value=50000)
    campaign_channel = st.selectbox("Campaign Channel", ["Social Media", "Email", "SEO", "PPC", "Referral"])
    campaign_type = st.selectbox("Campaign Type", ["Awareness", "Consideration", "Conversion", "Retention"])
    ad_spend = st.number_input("Ad Spend ($)", min_value=0.0, max_value=10000.0, value=200.0)

with col2:
    ctr = st.number_input("Click Through Rate (CTR)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    conversion_rate = st.number_input("Conversion Rate", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
    website_visits = st.number_input("Website Visits", min_value=0, max_value=500, value=10)
    pages_per_visit = st.number_input("Pages Per Visit", min_value=1.0, max_value=50.0, value=3.5, step=0.1)
    time_on_site = st.number_input("Time On Site (mins)", min_value=0.0, max_value=120.0, value=5.0, step=0.5)
    social_shares = st.number_input("Social Shares", min_value=0, max_value=100, value=2)

with col3:
    email_opens = st.number_input("Email Opens", min_value=0, max_value=50, value=1)
    email_clicks = st.number_input("Email Clicks", min_value=0, max_value=50, value=0)
    previous_purchases = st.number_input("Previous Purchases", min_value=0, max_value=50, value=2)
    loyalty_points = st.number_input("Loyalty Points", min_value=0, max_value=10000, value=150)
    advertising_platform = st.selectbox("Advertising Platform", ["Facebook", "Instagram", "Google", "Twitter", "LinkedIn"])
    advertising_tool = st.selectbox("Advertising Tool", ["Ads Manager", "Google Ads", "HubSpot", "Hootsuite"])

st.divider()
if st.button("🔮 Predict Conversion", use_container_width=True):
    try:
        with st.spinner("جاري التنبؤ..."):
            input_dict = {
                'CustomerID': customer_id,
                'Age': age,
                'Gender': gender,
                'Income': income,
                'CampaignChannel': campaign_channel,
                'CampaignType': campaign_type,
                'AdSpend': ad_spend,
                'ClickThroughRate': ctr,
                'ConversionRate': conversion_rate,
                'WebsiteVisits': website_visits,
                'PagesPerVisit': pages_per_visit,
                'TimeOnSite': time_on_site,
                'SocialShares': social_shares,
                'EmailOpens': email_opens,
                'EmailClicks': email_clicks,
                'PreviousPurchases': previous_purchases,
                'LoyaltyPoints': loyalty_points,
                'AdvertisingPlatform': advertising_platform,
                'AdvertisingTool': advertising_tool
            }

            input_df = pd.DataFrame([input_dict])

            # 🛠️ السطرين السحريين: تحويل أنواع البيانات أوتوماتيكياً
            for col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='ignore')

            # تشغيل التنبؤ بأمان تام
            results = predict_new_data(input_df)

            pred = results['Predicted_Conversion'].iloc[0]
            prob = results['Conversion_Probability'].iloc[0]

            st.subheader("📊 Prediction Result:")
            if pred == 1:
                st.success(f"✅ **High Potential Customer!**\n\nProbability of Conversion: **{prob * 100:.2f}%**")
            else:
                st.error(f"❌ **Low Potential Customer**\n\nProbability of Conversion: **{prob * 100:.2f}%**")

    except Exception as e:
        st.error(f"حدث خطأ أثناء التنبؤ: {e}")