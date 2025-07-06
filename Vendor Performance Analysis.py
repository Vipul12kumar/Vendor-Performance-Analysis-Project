#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import os
from sqlalchemy import create_engine


# In[14]:


# Replace the values below with your actual credentials
engine = create_engine('mysql+pymysql://root:your_password@127.0.0.1/inventory.db')


# In[15]:


def ingest_db(df,table_name, engine):
    df.to_sql(table_name,con=engine,if_exists='replace',index=False)


# In[ ]:


for file in os.listdir('data'):
    if '.csv' in file:
        df=pd.read_csv('data/'+file)
        print(df.shape)
        ingest_db(df,file[:-4],engine)


# In[18]:


import pandas as pd


# In[20]:


purchase=pd.read_csv("C:\\Users\\Jagveer singh\\Desktop\\inventory\\purchases.csv")


# In[21]:


purchase.head()


# In[23]:


purchase[purchase['VendorNumber']==4466]


# In[24]:


purchase_price=pd.read_csv("C:\\Users\\Jagveer singh\\Desktop\\inventory\\purchase_prices.csv")


# In[25]:


purchase_price.head(5)


# In[27]:


purchase_price[purchase_price["VendorNumber"]==4466]


# In[28]:


vendor_invoice=pd.read_csv("C:\\Users\\Jagveer singh\\Desktop\\inventory\\vendor_invoice.csv")


# In[29]:


vendor_invoice.head()


# In[31]:


vendor_invoice[vendor_invoice['VendorNumber']==4466]


# In[32]:


sales=pd.read_csv("C:\\Users\\Jagveer singh\\Desktop\\inventory\\sales.csv")


# In[33]:


sales.head()


# In[35]:


sales[sales["VendorNo"]==4466]


# In[39]:


purchase.groupby(['Brand','PurchasePrice'])[['Quantity','Dollars']].sum().head(3)


# In[40]:


purchase_price.head()


# In[41]:


vendor_invoice.head()


# In[43]:


vendor_invoice['PONumber'].nunique()


# In[44]:


vendor_invoice.shape


# In[47]:


sales.groupby('Brand')[['SalesDollars','SalesPrice','SalesQuantity']].sum().head()


# In[48]:


vendor_invoice.columns


# In[124]:


freight_summary=vendor_invoice.groupby('VendorNumber')[['Freight']].sum().rename(columns={'Freight': 'FreightCost'}).reset_index()


# In[123]:


freight_summary.head()


# In[58]:


purchase.head()


# In[60]:


purchase_price.head()


# In[64]:


filtered_purchase = purchase[purchase['PurchasePrice'] > 0]


# In[79]:


merged_df = pd.merge(
    purchase,
    purchase_price,
    on='Brand',
    how='inner'
)


# In[80]:


purchase[['VendorNumber','VendorName','Brand']]


# In[81]:


merged_df.head()


# In[105]:


df = merged_df[['VendorNumber_x', 'VendorName_x', 'Brand', 'Price', 'Volume', 'PurchasePrice_y']]


# In[106]:


df.head()


# In[107]:


dt = merged_df.groupby(['VendorNumber_x', 'VendorName_x', 'Brand'])[['Quantity', 'Dollars']].sum()\
       .rename(columns={'Quantity': 'TotalPurchaseQty', 'Dollars': 'TotalPurchaseDol'}).reset_index()


# In[98]:


dt


# In[108]:


final_df = pd.merge(df, dt, on=['VendorNumber_x', 'VendorName_x', 'Brand'], how='left')


# In[110]:


final_df = final_df[['VendorNumber_x', 'VendorName_x', 'Brand', 'Price', 'Volume', 'PurchasePrice_y',
                     'TotalPurchaseQty', 'TotalPurchaseDol']].rename(columns={'VendorNumber_x': 'VendorNumber', 'VendorName_x': 'VendorName','PurchasePrice_y':'ActualPrice'})


# In[114]:


final_df.tail()


# In[115]:


sales.head()


# In[141]:


a=sales.groupby(['VendorNo','Brand'])[['SalesDollars','SalesPrice','SalesQuantity','ExciseTax']].sum()\
       .rename(columns={'VendorNo':'VendorNumber','SalesDollars': 'TotalSalesDol', 'SalesPrice': 'TotalSalesPrice','SalesQuantity':'TotalSalesQty','ExciseTax':'TotalExciseTax'}).reset_index()


# In[168]:


a['Description']=sales['Description']


# In[169]:


data=pd.merge(final_df, freight_summary, on='VendorNumber',how='left')


# In[127]:





# In[170]:


final_table=pd.merge(data,a,on='Brand',how='left')


# In[171]:


final_table.head()


# In[172]:


final_table= final_table.drop('VendorNo', axis=1)


# In[173]:


final_table.head()


# In[185]:


final_table.shape


# In[186]:


final_table.nunique()


# In[187]:


final_table.columns


# In[188]:


Vendor_sales_summary= final_table.groupby(['VendorNumber', 'VendorName', 'Brand'], as_index=False).first()


# In[189]:


Vendor_sales_summary


# In[190]:


Vendor_sales_summary.dtypes


# In[191]:


Vendor_sales_summary.isnull().sum()


# In[192]:


Vendor_sales_summary['Volume']=Vendor_sales_summary['Volume'].astype('float64')


# In[193]:


Vendor_sales_summary.fillna(0,inplace=True)


# In[194]:


Vendor_sales_summary['VendorName']=Vendor_sales_summary['VendorName'].str.strip()


# In[195]:


Vendor_sales_summary.isnull().sum()


# In[196]:


Vendor_sales_summary.dtypes


# In[197]:


Vendor_sales_summary['Gross Profit']=Vendor_sales_summary['TotalSalesDol']-Vendor_sales_summary['TotalPurchaseDol']


# In[200]:


Vendor_sales_summary['ProfitMargin']= (Vendor_sales_summary['Gross Profit']/Vendor_sales_summary['TotalSalesDol'])*100


# In[201]:


Vendor_sales_summary


# In[202]:


Vendor_sales_summary['StockTurnover']=Vendor_sales_summary['TotalSalesQty']/Vendor_sales_summary['TotalPurchaseQty']


# In[204]:


Vendor_sales_summary['SalesPurchaseRatio']=Vendor_sales_summary['TotalSalesDol']/Vendor_sales_summary['TotalPurchaseDol']


# In[205]:


Vendor_sales_summary


# # Importing Libraries

# In[215]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sqlite3
from scipy.stats import ttest_ind
import scipy.stats as stats
warnings.filterwarnings('ignore')


# In[207]:


Vendor_sales_summary.head()


# # Exploratory Data Analysis
# 

# ### Previously, we examined the various tables in the database to identify key variables, understand their relationships, and determine which ones should be included in the final analysis.
# 

# ### In this phase of EDA, we will analyze the resultant table to gain insights into the distribution of each column. This will help us understand data patterns, identify anomalies, and ensure data quality before proceeding with further analysis.

# ### Summary Statistics

# In[209]:


Vendor_sales_summary.describe().T


# In[210]:


df=Vendor_sales_summary


# In[211]:


df


# In[216]:


#Distribution Plots for Numerical Columns
numerical_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize=(15,10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 4, i+1)# Adjust grid layout as needed 
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()


# In[217]:


# Outlier Detection with Boxplots
plt.figure(figsize=(15,10))
for i,col in enumerate(numerical_cols):
    plt.subplot(4,4,i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()


# # Summary Statistics Insights:

# ## Negative & Zero Values:

# #### Gross Profit: Minimum value is -52,002.78, indicating losses. Some products or transactions may be selling at a loss due to high costs or selling at discounts lower than the purchase price.
# 

# ##### Total Sales Quantity & Sales Dollars: Minimum values are 0, meaning some products were purchased but never sold. These could be slow-moving or obsolete stock.

# ##### Profit Margin: Has a minimum of -infinite, which suggests cases where revenue is zero or even lower than costs.

# ## Outliers Indicated by High Standard Deviations:

# ##### Purchase & Actual Prices: The max values (5,681.81 & 7,499.99) are significantly higher than the mean (24.39 & 35.64), indicating potential premium products.

# ##### Freight Cost: Huge variation, from 0.09 to 257,032.07, suggests logistics inefficiencies or bulk shipments.

# ##### Stock Turnover: Ranges from 0 to 274.5, implying some products sell extremely fast while others remain in stock indefinitely. Value more than 1 indicates that Sold quantity for that product is higher than purchased quantity due to either sales are being fulfilled from older stock.

# In[218]:


df=df[
    (df['Gross Profit'] > 0) &
    (df['ProfitMargin'] > 0) &
    (df['TotalSalesQty'] > 0)
]


# In[219]:


df


# In[220]:


#Distribution Plots for Numerical Columns
numerical_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize=(15,10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 4, i+1)# Adjust grid layout as needed 
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()


# In[221]:


#Count Plots for categorical columns
categorical_cols=["VendorName","Description"]

plt.figure(figsize=(12,5))
for i, col in enumerate(categorical_cols):
    plt.subplot(1,2,i+1)
    sns.countplot(y=df[col],order=df[col].value_counts().index[:10]) #Top 10 categories
    plt.title(f"Count Plot of {col}")
plt.tight_layout()
plt.show()


# In[222]:


#Correlation Heatmap
plt.figure(figsize=(12,8))
correlation_matrix=df[numerical_cols].corr()
sns.heatmap(correlation_matrix,annot=True,fmt=".2f",cmap="coolwarm",linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# # Correlation Insights

# ##### PurchasePrice has weak correlations with TotalSalesDollar(-0.012) and GrossProfit(-0.016), suggesting that price variations do not significantly impact sales revenue or profit

# ##### Strong Correlation between total purchase quantity and total sales quantity(0.999), confirming efficient inventory turnover

# ##### Negative correlation between profit margin & total sales price (-0.179) suggests that as sales prices increases, margins decrease, possibly due to competitive pricing pressure.

# ##### StockTurnover has weak negative correlations with both GrossProfit(-0.038) and ProfitMargin(-0.055), indicating that faster turnover does not necessarily result in higher profitability

# # Data Analysis

# #### Identify Brands that needs Promotional or pricing Adjustments which exihibit lower sales performance but highr profit margins.

# In[230]:


brand_performance=df.groupby('Description').agg({'TotalSalesDol':'sum', 'ProfitMargin':'mean'}).reset_index()


# In[232]:


low_sales_threshold=brand_performance['TotalSalesDol'].quantile(0.15)
high_margin_threshold=brand_performance['ProfitMargin'].quantile(0.85)


# In[233]:


low_sales_threshold


# In[235]:


high_margin_threshold


# In[238]:


#Filter brands with low sales but high profit margins
target_brands=brand_performance[(brand_performance ['TotalSalesDol'] <= low_sales_threshold) &
                                 (brand_performance ['ProfitMargin'] >= high_margin_threshold)]
print("Brands with Low Sales but High Profit Margins:")
display (target_brands.sort_values( 'TotalSalesDol'))


# In[254]:


brand_performance=brand_performance[brand_performance['TotalSalesDol']<10000]


# In[255]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=brand_performance,x='TotalSalesDol', y='ProfitMargin', color='Blue', label="All Brands", alpha=0.2)
sns.scatterplot(data=target_brands,x='TotalSalesDol', y='ProfitMargin', color='Red', label="Target Brands")
plt.axhline(high_margin_threshold, linestyle='--',color='black',label='High Margin Threshold')
plt.axhline(low_sales_threshold, linestyle='--',color='black',label='low sales Threshold')
plt.xlabel("Total Sales($)")
plt.ylabel("Profit Margin (%)")
plt.title("Brands for Promotional or pricing Adjustments")
plt.legend()
plt.grid(True)
plt.show()


# #### which vendor and brands demonstrate the highest sales performance?

# In[256]:


top_vendors=df.groupby("VendorName")["TotalSalesDol"].sum().nlargest(10)
top_brands=df.groupby("Description")["TotalSalesDol"].sum().nlargest(10)


# In[257]:


top_vendors


# In[258]:


top_brands


# In[261]:


def format_dollars(value):
    if value>=1_000_000:
        return f"{value/1_000_000:2f}M"
    elif value>=1_000:
         return f"{value/1_000:2f}K"
    else:
        return str(value)
        


# In[265]:


plt.figure(figsize=(15,5))

#Plot for Top Vendors
plt.subplot(1,2,1)
ax1=sns.barplot(y=top_vendors.index,x=top_vendors.values,palette="Blues_r")
plt.title("Top 10 Vendors by sales")

for bar in ax1.patches:
    ax1.text(bar.get_width()+(bar.get_width()*0.02),
             bar.get_y()+bar.get_height()/2,
             format_dollars(bar.get_width()),
             ha='left', va='center',fontsize=10,color='black')
#Plot for top brands
plt.subplot(1,2,2)
ax2=sns.barplot(y=top_brands.index.astype(str),x=top_brands.values,palette="Reds_r")
plt.title("Top 10 Brands by Sales")
for bar in ax2.patches:
    ax2.text(bar.get_width() + (bar.get_width()*0.02),
             bar.get_y()+ bar.get_height()/2,
             format_dollars(bar.get_width()),
             ha='left', va='center',fontsize=10,color='black')
plt.tight_layout()
plt.show()


# ##### Which Vendor contribute the most to total purchases dollars?

# In[281]:


Vendor_Performance=df.groupby('VendorName').agg({
    'TotalPurchaseDol':'sum',
    'Gross Profit':'sum',
    'TotalSalesDol':'sum'}).reset_index()


# In[296]:


Vendor_Performance['PurchaseContribution%']=Vendor_Performance['TotalPurchaseDol']/Vendor_Performance['TotalPurchaseDol'].sum()*100


# In[297]:


Vendor_Performance=round(Vendor_Performance.sort_values('PurchaseContribution%', ascending=False),2)


# In[298]:


#Display Top 10 Vendors
top_vendors=Vendor_Performance.head(10)
top_vendors['TotalSalesDol']=top_vendors['TotalSalesDol'].apply(format_dollars)
top_vendors['TotalPurchaseDol']=top_vendors['TotalPurchaseDol'].apply(format_dollars)
top_vendors['Gross Profit']=top_vendors['Gross Profit'].apply(format_dollars)
top_vendors


# In[299]:


top_vendors['PurchaseContribution%'].sum()


# In[300]:


top_vendors['Cumulative_Contribution%']=top_vendors['PurchaseContribution%'].cumsum()


# In[301]:


top_vendors


# In[302]:


fig, ax1 = plt.subplots(figsize=(10,6))

#Bar plot for Purchase Contribution%

sns.barplot(x=top_vendors['VendorName'], y=top_vendors['PurchaseContribution%'], palette="mako", ax=ax1)

for i, value in enumerate(top_vendors['PurchaseContribution%']):
    ax1.text(i, value - 1, str(value)+'%', ha='center', fontsize=10, color='white')

# Line Plot for Cumulative Contribution%

ax2 = ax1.twinx()
ax2.plot(top_vendors ['VendorName'], top_vendors['Cumulative_Contribution%'], color='red', marker='o', linestyle='dashed', label='Cumulative')

ax1.set_xticklabels (top_vendors ['VendorName'], rotation=90)

ax1.set_ylabel('PurchasContribution %', color='blue')

ax2.set_ylabel('Cumulative Contribution %', color='red')

ax1.set_xlabel('Vendors')

ax1.set_title('Pareto Chart: Vendor Contribution to Total Purchases')

ax2.axhline(y=100, color='gray', linestyle='dashed', alpha=0.7)

ax2.legend(loc='upper right')

plt.show()


# ##### How much of total procurement is dependent on the top Vendors?

# In[305]:


print(f"Total Purchase Contribution of total 10 vendor is {round(top_vendors['PurchaseContribution%'].sum(),2)}%")


# In[310]:


vendors = list(top_vendors['VendorName'].values)
purchase_contributions = list(top_vendors['PurchaseContribution%'].values) 
total_contribution = sum(purchase_contributions)
remaining_contribution = 100 -total_contribution

# Append "Other Vendors" category

vendors.append("Other Vendors")
purchase_contributions.append(remaining_contribution)

# Donut Chart
fig, ax = plt.subplots (figsize=(8, 8))
wedges, texts, autotexts = ax.pie(purchase_contributions, labels =vendors, autopct='%1.1f%%',
                                  startangle=140, pctdistance=0.85, colors=plt.cm.Paired.colors)

# Draw a white circle in the center to create a "donut" effect
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

# Add Total Contribution annotation in the center
plt.text(0, 0, f"Top 10 Total:\n{total_contribution:.2f}%", fontsize=14, fontweight='bold', ha='center')

plt.title("Top 10 Vendor's Purchase Contribution (%)")
plt.show()


# ##### Does purchasing in bulk reduce the unit price, and what is the optimal purchase volume for cost savings?

# In[311]:


df['UnitPurchasePrice']=df['TotalPurchaseDol']/df['TotalPurchaseQty']


# In[313]:


df['OrderSize']=pd.qcut(df["TotalPurchaseQty"],q=3,labels=["Small","Medium","Large"])


# In[316]:


df.groupby('OrderSize')[['UnitPurchasePrice']].mean()


# In[317]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="OrderSize", y="UnitPurchasePrice", palette="Set2")
plt.xlabel("Order Size")
plt.title("Impact of Bulk Purchasing on Unit Price")
plt.ylabel("Average Unit Purchase Price")
plt.show()


# ##### Vendors buying in bulk (Large Order Size) get the lowest unit price ($10.78 per unit), meaning higher margins if they can manage inventory efficiently.

# ##### The price difference between Small and Large orders is substantial (~72% reduction in unit cost)

# ##### This suggests that bulk pricing strategies successfully encourage vendors to purchase in larger volumes, leading to higher overall sales despite lower per-unit revenue.

# #### Which vendor have low inventory turnover, indicating excess stock and slow-moving products?

# In[320]:


df[df['StockTurnover']<1].groupby('VendorName')[['StockTurnover']].mean().sort_values('StockTurnover', ascending=True).head(10)


# #### How much capital is locked in unsold inventory per vendor, and which vendors contribute the most to it?

# In[324]:


df ["UnsoldInventoryValue"] = (df["TotalPurchaseQty"] - df ["TotalSalesQty"]) * df ["ActualPrice"]
print('Total Unsold Capital:', format_dollars (df ["UnsoldInventoryValue"].sum()))


# In[322]:


df.columns


# In[326]:


# Aggregate Capital Locked per Vendor
inventory_value_per_vendor = df.groupby("VendorName") ["UnsoldInventoryValue"].sum().reset_index()

# Sort Vendors with the Highest Locked Capital
inventory_value_per_vendor = inventory_value_per_vendor.sort_values(by="UnsoldInventoryValue", ascending=False)
inventory_value_per_vendor['UnsoldInventoryValue'] = inventory_value_per_vendor ['UnsoldInventoryValue'].apply(format_dollars)
inventory_value_per_vendor.head(10)


# ##### What is the 95% confidence intervals for profit margins of top-performing and low-performing vendors.

# In[329]:


top_threshold = df ["TotalSalesDol"].quantile(0.75)
low_threshold = df ["TotalSalesDol"].quantile(0.25)


# In[330]:


top_vendors = df [df ["TotalSalesDol"] >= top_threshold] ["ProfitMargin"].dropna()
low_vendors = df [df ["TotalSalesDol"] <= low_threshold] ["ProfitMargin"].dropna()


# In[331]:


top_vendors


# In[343]:


def confidence_interval(data, confidence=0.95):
     mean_val = np.mean(data)
     std_err = np.std(data, ddof=1) / np.sqrt(len(data)) 
     t_critical= stats.t.ppf((1 + confidence) / 2, df=len(data) - 1)
     margin_of_error = t_critical * std_err
     return mean_val, mean_val - margin_of_error, mean_val + margin_of_error


# In[344]:


top_mean,top_lower,top_upper = confidence_interval(top_vendors)
low_mean,low_lower,low_upper = confidence_interval(low_vendors)

print(f"Top Vendors 95% CI: ({top_lower:.2f}, {top_upper:.2f}), Mean: {top_mean:.2f}")
print(f"Low Vendors 95% CI: ({low_lower:.2f}, {low_upper:.2f}), Mean: {low_mean:.2f}")
plt.figure(figsize=(12, 6))

# Top Vendors Plot
sns.histplot(top_vendors, kde=True, color="blue", bins= 30, alpha=0.5, label="Top Vendors")
plt.axvline(top_lower, color="blue", linestyle="--", label=f"Top Lower: {top_lower:.2f}")
plt.axvline(top_upper, color="blue", linestyle="--", label=f"Top Upper: {top_upper:.2f}")
plt.axvline(top_mean, color="blue", linestyle="-", label=f"Top Mean: {top_mean:.2f}")

# Low Vendors Plot
sns.histplot(low_vendors, kde=True, color="red", bins= 30, alpha=0.5, label="Low Vendors")
plt.axvline(low_lower, color="red", linestyle="-", label=f"Low Lower: {low_lower:.2f}")
plt.axvline(low_upper, color="red", linestyle="--", label=f"Low Upper: {low_upper:.2f}")
plt.axvline(low_mean, color="red", linestyle="-", label=f"Low Mean: {low_mean:.2f}")

# Finalize Plot
plt.title("Confidence Interval Comparison: Top vs. Low Vendors (Profit Margin)")
plt.xlabel("Profit Margin (%)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()


# #### The confidence interval for low-performing vendors (40.48% to 42.62%) is significantly higher than that of top-performing vendors (30.74% to 31.61%).

# #### This suggests that vendors with lower sales tend to maintain higher profit margins, potentially due to premium pricing or lower operational costs

# #### For High-Performing Vendors: If they aim to improve profitability, they could explore selective price adjustments, cost optimization, or bundling strategies.

# #### For Low-Performing Vendors: Despite higher margins, their low sales volume might indicate a need for better marketing, competitive pricing, or improved distribution strategies.

# ### Is there a significant difference in profit margins between top-performing and low-performing vendors?

# ## Hypothesis:

# ### Ho (Null Hypothesis): There is no significant difference in the mean profit margins of top-performing and low-performing vendors.

# #### H₁ (Alternative Hypothesis): The mean profit margins of top-performing and low-performing vendors are significantly different.

# In[348]:


top_threshold= df ["TotalSalesDol"].quantile(0.75)
low_threshold = df ["TotalSalesDol"].quantile(0.25)

top_vendors = df [df ["TotalSalesDol"] >= top_threshold] ["ProfitMargin"].dropna()
low_vendors = df [df ["TotalSalesDol"] <= low_threshold] ["ProfitMargin"].dropna()

# Perform Two-Sample T-Test
t_stat, p_value = ttest_ind(top_vendors, low_vendors, equal_var=False)

#Print results
print(f"T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")
                                             
if p_value < 0.05:
   print("Reject Ho: There is a significant difference in profit margins between top and low-performing vendors.")
else:
   print("Fail to Reject Ho: No significant difference in profit margins.")


# In[5]:


get_ipython().system('jupyter nbconvert your_notebook.ipynb --to webpdf --allow-chromium-download')



# In[ ]:




