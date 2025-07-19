import streamlit as st
import pickle
import numpy as np
import pandas as pd
from preprocessing import CustomPreprocessor 
import time

                
with open('xgb_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)


# Load the trained pipeline
st.sidebar.header("📊 E-Commerce Customer Churn Predictor",width=10000)
st.sidebar.info("this app predicts the customer churn from an e-commerce application based on your inputs")
st.sidebar.image("441-4411049_ecommerce-website-design-icon-hd-png-download.png")
data=pd.read_excel("E Commerce Dataset.xlsx",sheet_name=1)
data2=pd.read_excel("E Commerce Dataset.xlsx",sheet_name=0)


opt=st.sidebar.radio("**GO TO**",["Data","Customer Profile",'Prediction'])

if opt=='Data':
    st.header("DATA")
    st.image("image copy 2.png",width=150)
    st.info("The data set belongs to a leading online E-Commerce company. An online retail (E commerce) company wants to know the customers who are going to churn, so accordingly they can approach customer to offer some promos.")
    with st.expander("DATA"):
      st.dataframe(data,height=250)
    with st.expander("FEATURE DESCRIPTION"):
       st.dataframe(data2)
    with st.expander("CHARACTERSTICS OF DATA"):
      st.dataframe(data.describe(include='all').T)

if opt=='Customer Profile':
   st.header("Customer Profile Input")
   st.image("image copy.png",width=100)
   st.info("Please fill in the customer details below to generate a churn prediction")
   tenure=st.slider("Tenure",0,65,10)
   p_log_device=st.selectbox("Preferred Login Device",['Mobile Phone','Computer'])
   city_tier=st.selectbox("City Tier",["1",'2','3'])
   ware_to_home=st.slider("WarehouseToHome",2,30,15)
   p_pay_method=st.selectbox("Preffered Payment Mode",['Debit Card','UPI','Credit Card','Cash on Delivery','E wallet'])
   gender=st.selectbox("Gender",['Male','Female'])
   hours=st.slider("Hours spent on App",0,8,3)
   no_of_dev=st.slider("Number of Devices Registered",1,8,4)
   order=st.selectbox("Preffered Order Category",['Laptop & Accessory','Mobile Phone','Fashion','Grocery','Others'])
   sat_score=st.selectbox('Satisfaction Score',[1,2,3,4,5])
   mar_status=st.selectbox("Marital Status",['Married','Single','Divorced'])
   no_of_add=st.slider("Number of Addresses",1,25,5)
   complain=st.selectbox("Complain",[0,1])
   order_hike=st.slider("Order Amount Hike From Last Year",8,30,15)
   coupons_used=st.slider("Coupons used",0,20,3)
   ordercount=st.slider("Order Count",1,16,3)
   day_last_order=st.slider("Days Since Last Order",0,50,5)
   cashback=st.slider("Cashback Amount",0.00,325.00,178.00)
   
   if st.button("Predict"):
    df={'Tenure':tenure,
          'PreferredLoginDevice':p_log_device,
          'CityTier':city_tier,
          'WarehouseToHome':ware_to_home,
          'PreferredPaymentMode':p_pay_method,
          'Gender':gender,
          'HourSpendOnApp':hours,
          'NumberOfDeviceRegistered':no_of_dev,
          'PreferedOrderCat':order,
          'SatisfactionScore':sat_score,
          'MaritalStatus':mar_status,
          'NumberOfAddress':no_of_add,
          'Complain':complain,
          'OrderAmountHikeFromlastYear':order_hike, # type: ignore
          'CouponUsed':coupons_used,
          'OrderCount':ordercount,
          'DaySinceLastOrder':day_last_order,
          'CashbackAmount':cashback}
    st.session_state['user_input'] = df
    st.success("✅ Input successfully captured!")
    st.write("Now go to **Prediction** in the sidebar to view the customer's churn probability.")

if opt=='Prediction':
    st.header("Churn Prediction Result")
    st.info("""
This section analyzes the provided customer profile and evaluates the **likelihood of churn**""")
    st.subheader("Customer Summary")
    if 'user_input' in st.session_state:
       final = pd.DataFrame([st.session_state['user_input']])
       st.dataframe(final)
       prediction = model.predict(final)[0]
       probability=model.predict_proba(final)[0]
       churn_prob=round(probability[1]*100,2)
       if churn_prob < 30:
        label = "No Churn"
        message = "This customer is considered loyal and stable. Engagement is strong."
       elif 30 <= churn_prob <= 60:
        label = "Uncertain"
        message = "This customer shows signs of disengagement. You should monitor behavior and offer personalized retention strategies."
       else:
        label = "Churn"
        message = "High risk of churn detected. Immediate retention action is recommended — reach out with exclusive offers or loyalty programs."
       

       # Create DataFrame to display
       result_df = pd.DataFrame({
        'Prediction': [label],
        'Probability (No Churn)': [round(probability[0]*100, 2)],
        'Probability (Churn)': [round(probability[1]*100, 2)]
      })

       # Display
       st.subheader("📊 Prediction Result")
       st.dataframe(
        result_df,
        column_config={
            "Prediction": st.column_config.TextColumn("Customer Status"),
            "Probability (No Churn)": st.column_config.ProgressColumn(
                "No Churn", format="%.2f%%", min_value=0, max_value=100
            ),
            "Probability (Churn)": st.column_config.ProgressColumn(
                "Churn", format="%.2f%%", min_value=0, max_value=100
            )
        },
        use_container_width=True,
        hide_index=True
       )
       if churn_prob < 30:
          with st.spinner("Analyzing..."):
           time.sleep(3)  # Simulated wait for effect
          st.success("🎉 Great news! This customer is **not likely to churn**.")
          st.balloons()
       elif 30<churn_prob<60:
         st.warning("act fast!!,this customer is more likely to churn")
       else:
         st.info("Keep an eye on this customer. Their behavior could shift either way")
         
        
    else:
     st.warning("⚠️ Please fill in the customer details and click 'Predict' first.")

