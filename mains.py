import streamlit as st
import pandas as pd
import plotly.express as px
import os

def func():
    st.set_page_config(page_title="Inventory Dashboard", layout="wide")
    st.title("Inventory Dashboard")
    file_path = r'C:\Users\aksps\OneDrive\Python\Jackfruit\items.xlsx'

    if os.path.exists(file_path):
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
                
            with open(r'C:\Users\aksps\OneDrive\Python\Jackfruit\items.xlsx', "rb") as file:
                btn = st.download_button(
                    label="Excel File",
                    data=file,
                    file_name=r':\Users\aksps\OneDrive\Python\Jackfruit\items.xlsx',
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
             
            df.columns = df.columns.str.strip()

            if 'PARTICULARS' in df.columns:
                df_clean = df.dropna(subset=['PARTICULARS']).copy()
            else:
                st.error("Column 'PARTICULARS' not found. Please check your file headers.")
                st.stop()
                
            numeric_cols = ['QUANTITY', 'AMOUNT', 'EXISTING QUANTITY', 'REMAINING QUANTITY']
            
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

            if 'EXISTING QUANTITY' in df_clean.columns and 'QUANTITY' in df_clean.columns:
                df_clean['PERCENT_SOLD'] = (df_clean['QUANTITY'] / df_clean['EXISTING QUANTITY']) * 100
            
            st.divider()
            
            st.subheader("Overview")
            total_revenue = df_clean['AMOUNT'].sum() if 'AMOUNT' in df_clean.columns else 0
            total_sold = df_clean['QUANTITY'].sum() if 'QUANTITY' in df_clean.columns else 0

            if not df_clean.empty and 'QUANTITY' in df_clean.columns:
                most_sold_item = df_clean.loc[df_clean['QUANTITY'].idxmax()]
                least_sold_item = df_clean.loc[df_clean['QUANTITY'].idxmin()]
                most_sold_str = f"{most_sold_item['PARTICULARS']}"
                least_sold_str = f"{least_sold_item['PARTICULARS']}"
                most_sold_val = f"{int(most_sold_item['QUANTITY'])} units"
                least_sold_val = f"{int(least_sold_item['QUANTITY'])} units"
            else:
                most_sold_str, least_sold_str = "N/A", "N/A"
                most_sold_val, least_sold_val = "0", "0"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Revenue", f"₹{total_revenue:,.0f}") 
            c2.metric("Units Sold", f"{int(total_sold)}")
            c3.metric("Most Sold", most_sold_str, most_sold_val)
            c4.metric("Least Sold", least_sold_str, least_sold_val)

            st.divider()

            st.subheader("Sales Distribution")
            
            r1c1, r1c2 = st.columns(2)
            
            with r1c1:
                st.markdown("Sales Volume")
                if 'QUANTITY' in df_clean.columns:
                    fig1 = px.pie(df_clean, values='QUANTITY', names='PARTICULARS', hole=0.5,
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig1.update_traces(textposition='inside', textinfo='percent+label')
                    fig1.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig1, use_container_width=True)
                
            with r1c2:
                st.markdown("Revenue Share")
                if 'AMOUNT' in df_clean.columns:
                    fig2 = px.pie(df_clean, values='AMOUNT', names='PARTICULARS', hole=0.5,
                                color_discrete_sequence=px.colors.qualitative.Pastel2)
                    fig2.update_traces(textposition='inside', textinfo='percent')
                    fig2.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")
            
            bar_col1, bar_col2 = st.columns(2)
            
            with bar_col1:
                st.markdown("Stock Comparison")
                if 'EXISTING QUANTITY' in df_clean.columns and 'REMAINING QUANTITY' in df_clean.columns:
                    df_melted = df_clean.melt(id_vars='PARTICULARS', 
                                            value_vars=['EXISTING QUANTITY', 'REMAINING QUANTITY'], 
                                            var_name='Status', value_name='Count')
                    df_melted['Status'] = df_melted['Status'].str.replace('QUANTITY', '').str.title().str.strip()
                    fig3 = px.bar(df_melted, x='Count', y='PARTICULARS', color='Status', barmode='group',
                                orientation='h',  # Horizontal orientation
                                color_discrete_map={'Existing': '#636EFA', 'Remaining': '#EF553B'},
                                height=500) # Fixed height for scrollable area if needed
                    
                    fig3.update_layout(yaxis={'categoryorder':'total ascending'}, 
                                    xaxis_title="Quantity", yaxis_title=None,
                                    legend_title=None, margin=dict(l=0))
                    st.plotly_chart(fig3, use_container_width=True)

            with bar_col2:
                st.markdown("Inventory Turnover (%)")
                if 'PERCENT_SOLD' in df_clean.columns:
                    df_sorted = df_clean.sort_values(by='PERCENT_SOLD', ascending=True)
                    
                    fig4 = px.bar(df_sorted, x='PERCENT_SOLD', y='PARTICULARS', 
                                orientation='h',
                                text_auto='.1f',
                                color='PERCENT_SOLD',
                                color_continuous_scale='Blues',
                                height=500)
                    
                    fig4.update_layout(xaxis_title="% Sold", yaxis_title=None,
                                    xaxis=dict(range=[0, 100]),
                                    coloraxis_showscale=False,
                                    margin=dict(l=0))
                    st.plotly_chart(fig4, use_container_width=True)

            st.divider()

            with st.expander("Detailed Data Table"):
                format_dict = {
                    'PERCENT_SOLD': "{:.1f}%", 'QUANTITY': "{:.0f}", 'AMOUNT': "{:.0f}",
                    'EXISTING QUANTITY': "{:.0f}", 'REMAINING QUANTITY': "{:.0f}"
                }
                valid_formats = {k: v for k, v in format_dict.items() if k in df_clean.columns}
                
                def highlight_low(val):
                    return 'color: red; font-weight: bold' if val == 'LOW STOCK' else 'color: green; font-weight: bold'

                st_df = df_clean.style.format(valid_formats)
                if 'STATUS' in df_clean.columns:
                    st_df = st_df.applymap(highlight_low, subset=['STATUS'])
                    
                st.dataframe(st_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")

    else:
        st.error(f"File not found at path: {file_path}")
        st.info("Please check if the file path is correct and the file exists.")

if __name__ == "__main__":
    func()