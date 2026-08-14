import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import yfinance as yf

import os
from google import genai

def get_gemini_api_key():
  """Checks Streamlit secrets, Colab userdata, or OS environment variables for GEMINI_API_KEY."""
  # 1. Try Streamlit Secrets (for Streamlit Cloud)
  if "GEMINI_API_KEY" in st.secrets:
    return st.secrets["GEMINI_API_KEY"]

  # 2. Try Google Colab Userdata (if running inside Colab)
  try:
    from google.colab import userdata

    return userdata.get("GEMINI_API_KEY")
  except ImportError:
    pass

  # 3. Fallback to standard environment variable
  return os.environ.get("GEMINI_API_KEY")


def fetch_equity_news_summary(stock_ticker):
  """Uses Gemini Flash Latest to generate a short, up-to-date financial context/news summary for the ticker."""
  api_key = get_gemini_api_key()

  if not api_key:
    return "⚠️ *GEMINI_API_KEY missing. Add GEMINI_API_KEY to your Streamlit secrets or environment variables to enable news context.*"

  try:
    client = genai.Client(api_key=api_key)

    prompt = f"""
        Provide a concise, 1-paragraph (3-4 sentences max) overview of recent major news, financial catalysts, 
        or business developments driving the stock performance for ticker symbol '{stock_ticker}'. 
        Keep the tone professional, objective, and financial-focused.
        """

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
    )

    return response.text
  except Exception as e:
    return f"Unable to fetch news summary for {stock_ticker}. Error: {e}"

# --- Page Configuration ---
st.set_page_config(
    page_title="CAPM Beta Analyzer", page_icon="📈", layout="wide"
)

st.title("📈 Stock Beta & CAPM Regression Analyzer")






st.markdown(
    "Enter any equity ticker symbol below to analyze its 60-month risk/return"
    " profile against the **S&P 500 (^GSPC)**."
)

# --- Sidebar Inputs ---
stock_ticker = st.text_input("Enter Stock Ticker Symbol:", "NVDA").upper()
market_ticker = "^GSPC"

if st.button("Run Analysis", type="primary"):
  with st.spinner(f"Fetching 60 months of data for {stock_ticker}..."):
    try:
      # 1. Download Price Data
      tickers = [stock_ticker, market_ticker]
      df_prices = yf.download(tickers, period="61mo", interval="1mo")[
          "Close"
      ].dropna()

      # 2. Calculate Monthly Percentage Returns
      df_returns = df_prices.pct_change().dropna().tail(60)

      # Extract X (Market) and Y (Stock)
      X = df_returns[market_ticker]
      Y = df_returns[stock_ticker]

      # Calculate Summary Stats
      X_mean, X_std = X.mean(), X.std()
      Y_mean, Y_std = Y.mean(), Y.std()

      # 3. Fit OLS Regression
      X_with_const = sm.add_constant(X)
      model = sm.OLS(Y, X_with_const)
      results = model.fit()

      # 4. Extract Key Business Statistics
      beta_0 = results.params["const"]
      beta_1 = results.params[market_ticker]
      r_squared = results.rsquared
      r = np.sqrt(r_squared) * np.sign(beta_1)
      p_value = results.pvalues[market_ticker]

      # --- METRICS DISPLAY ---
      st.subheader("📊 Key Regression Output")
      col1, col2, col3, col4 = st.columns(4)
      col1.metric("Beta (Slope, β₁)", f"{beta_1:.4f}")
      col2.metric("R-Squared (R²)", f"{r_squared:.2%}")
      col3.metric("Correlation (r)", f"{r:.4f}")
      col4.metric("p-value", f"{p_value:.4e}")

      # --- RECENT NEWS & CONTEXT BLOCK ---
      st.markdown(f"### 📰 Financial Context & News: {stock_ticker}")
      with st.spinner(f"Fetching news context for {stock_ticker}..."):
        news_summary = fetch_equity_news_summary(stock_ticker)
        st.write(news_summary)
        
        
        
        # --- DYNAMIC EXECUTIVE SUMMARY ---
      if beta_1 > 1.2:
        volatility_type = (
            "high-beta / aggressive stock (significantly higher market"
            " volatility)"
        )
      elif beta_1 < 0.8:
        volatility_type = (
            "low-beta / defensive stock (significantly lower market volatility)"
        )
      else:
        volatility_type = (
            "moderate-beta stock (moves closely with broad market volatility)"
        )

      risk_comp = "higher total risk" if Y_std > X_std else "lower total risk"

      if p_value < 0.05:
        sig_text = (
            f"Statistically significant relationship (p = {p_value:.4e} <"
            f" 0.05). Market returns are a valid predictor of {stock_ticker}."
        )
      else:
        sig_text = (
            f"Not statistically significant (p = {p_value:.4f} >= 0.05). S&P"
            f" 500 returns do not reliably predict {stock_ticker}."
        )

      st.markdown("### 📝 Executive Summary")
      st.info(f"""
            **• Risk & Return Profile:**  
            {stock_ticker} averaged a monthly return of **{Y_mean:.2%}** (Std Dev: **{Y_std:.2%}**), compared to the S&P 500 average of **{X_mean:.2%}** (Std Dev: **{X_std:.2%}**). Overall, {stock_ticker} exhibits **{risk_comp}** than the benchmark index.

            **• Market Sensitivity (Beta & R²):**  
            With a Beta of **{beta_1:.4f}**, {stock_ticker} is categorized as a **{volatility_type}**. An R-squared of **{r_squared:.2%}** indicates that **{r_squared*100:.1f}%** of {stock_ticker}'s return variance is explained strictly by broad market movements.

            **• Hypothesis Testing:**  
            {sig_text}
            """)

      # --- VISUALIZATION & DATA TABLE ---
      tab1, tab2, tab3 = st.tabs(["📉 Scatter & Regression Plot", "📋 Return Data", "📉 Returns Plot"])

      with tab1:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(
            X * 100,
            Y * 100,
            alpha=0.7,
            color="navy",
            label=f"Monthly Returns ({stock_ticker})",
        )
        x_vals = np.linspace(X.min(), X.max(), 100)
        y_vals = (beta_0 + beta_1 * x_vals) * 100
        ax.plot(
            x_vals * 100,
            y_vals,
            color="red",
            linewidth=2,
            label=f"Regression Line (Beta = {beta_1:.2f})",
        )
        ax.set_title(
            f"CAPM Beta Regression: {stock_ticker} vs. S&P 500 (Last 60"
            " Months)",
            fontweight="bold",
        )
        ax.set_xlabel("S&P 500 Monthly Return (%)")
        ax.set_ylabel(f"{stock_ticker} Monthly Return (%)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        st.pyplot(fig)

      with tab2:
        df_excel = pd.DataFrame({
            "date": df_returns.index.strftime("%Y-%m"),
            "sp500 close": df_prices.loc[df_returns.index, market_ticker].round(
                2
            ),
            "equity close": df_prices.loc[df_returns.index, stock_ticker].round(
                2
            ),
            "sp500 return": df_returns[market_ticker].round(4),
            "equity return": df_returns[stock_ticker].round(4),
        })

        st.dataframe(df_excel, use_container_width=True)

        # Excel Download Button
        excel_buffer = pd.ExcelWriter(
            "equity_datafile.xlsx", engine="openpyxl"
        )
        df_excel.to_excel(excel_buffer, index=False)
        excel_buffer.close()

        with open("equity_datafile.xlsx", "rb") as f:
          st.download_button(
              label="📥 Download Excel Dataset",
              data=f,
              file_name=f"{stock_ticker}_vs_SP500_60M.xlsx",
              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          )

    except Exception as e:
      st.error(
          f"Could not retrieve data for **{stock_ticker}**. Please verify the"
          f" ticker symbol. Error details: {e}"
      )
