import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, chi2
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VKC Probit Analysis",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CALCULATION ENGINE (Iterative Finney Method) ---
def probit_analysis(df):
    # Clean and Map Columns
    df.columns = df.columns.str.strip().str.lower()
    cols = df.columns
    try:
        c_dose = next((c for c in cols if 'dose' in c or 'conc' in c), cols[0])
        c_total = next((c for c in cols if 'total' in c or 'n' == c), cols[1])
        c_dead = next((c for c in cols if 'dead' in c or 'kill' in c), cols[2])
    except:
        return None, "Error: Could not identify Dose, Total, and Dead columns."

    # Filter Data
    df_clean = df[pd.to_numeric(df[c_dose], errors='coerce') > 0].copy()
    doses = df_clean[c_dose].values.astype(float)
    n_subj = df_clean[c_total].values.astype(float)
    r_dead = df_clean[c_dead].values.astype(float)
    x_log = np.log10(doses)

    # Initial Estimates
    p_obs = r_dead / n_subj
    p_safe = np.clip(p_obs, 0.01, 0.99)
    y_emp = norm.ppf(p_safe) + 5
    slope, intercept = np.polyfit(x_log, y_emp, 1)

    final_vars = {}
    r_squared = 0.0

    # Finney Iteration Loop
    for i in range(25):
        old_slope, old_intercept = slope, intercept
        
        Y_exp = intercept + slope * x_log
        P_exp = norm.cdf(Y_exp - 5)
        P_exp = np.clip(P_exp, 1e-10, 1.0 - 1e-10)
        
        Z_val = norm.pdf(norm.ppf(P_exp))
        weight = (n_subj * Z_val**2) / (P_exp * (1 - P_exp))
        y_work = Y_exp + (p_obs - P_exp) / Z_val
        
        sw = np.sum(weight)
        swx = np.sum(weight * x_log)
        swy = np.sum(weight * y_work)
        swxx = np.sum(weight * x_log**2)
        swxy = np.sum(weight * x_log * y_work)
        swyy = np.sum(weight * y_work**2) # Needed for R-Squared
        
        x_bar = swx / sw
        y_bar = swy / sw
        
        Sxx = swxx - (swx**2 / sw)
        Sxy = swxy - (swx * swy / sw)
        Syy = swyy - (swy**2 / sw) # Total Sum of Squares
        
        slope = Sxy / Sxx
        intercept = y_bar - slope * x_bar
        
        if abs(slope - old_slope) < 1e-6 and abs(intercept - old_intercept) < 1e-6:
            # Calculate R-Squared
            if Sxx * Syy != 0:
                r_squared = (Sxy**2) / (Sxx * Syy)
            
            final_vars = {
                'doses': doses, 'n': n_subj, 'r': r_dead, 'x_log': x_log,
                'p_obs': p_obs, 'y_emp': norm.ppf(np.clip(p_obs, 0.001, 0.999))+5,
                'y_exp': Y_exp, 'y_work': y_work, 'w': weight,
                'wx': weight*x_log, 'wy': weight*y_work, 'wxy': weight*x_log*y_work,
                'sw': sw, 'sxx': Sxx, 'x_bar': x_bar
            }
            break
            
    # Stats
    ld50_log = (5 - intercept) / slope
    ld50 = 10 ** ld50_log
    
    P_final = norm.cdf(final_vars['y_exp'] - 5)
    r_exp = n_subj * P_final
    var_exp = n_subj * P_final * (1 - P_final)
    var_exp[var_exp<1e-9] = 1e-9
    chi_sq = np.sum(((r_dead - r_exp)**2) / var_exp)
    df_deg = len(doses) - 2
    h_factor = chi_sq / df_deg if (chi_sq > df_deg and df_deg > 0) else 1.0

    # Build 15-Column Table
    table_data = []
    for i in range(len(doses)):
        table_data.append({
            '1. Dose': doses[i], '2. N': int(n_subj[i]), '3. R': int(r_dead[i]),
            '4. LogD': x_log[i], '5. %Obs': p_obs[i]*100, 
            '6. EmpPr': final_vars['y_emp'][i], '7. ExpPr': final_vars['y_exp'][i],
            '11. WorkPr': final_vars['y_work'][i], '12. W': final_vars['w'][i],
            '13. WX': final_vars['wx'][i], '14. WY': final_vars['wy'][i], 
            '15. WXY': final_vars['wxy'][i]
        })
    df_table = pd.DataFrame(table_data)

    # Build LD Table
    ld_results = []
    t_val = 1.96 if df_deg <= 0 else t.ppf(0.975, df_deg)
    var_slope = (1/final_vars['sxx']) * h_factor
    g = (t_val**2 * var_slope) / (slope**2)
    
    for val in [10, 25, 50, 90, 95, 99]:
        y_target = norm.ppf(val/100) + 5
        m = (y_target - intercept) / slope
        dose_val = 10**m
        
        if g < 1:
            term1 = m + (g/(1-g))*(m - final_vars['x_bar'])
            term2 = (t_val/(slope*(1-g))) * np.sqrt((1-g)/final_vars['sw'] + ((m-final_vars['x_bar'])**2/final_vars['sxx'])*h_factor)
            lower = 10**(term1 - term2)
            upper = 10**(term1 + term2)
        else:
            lower, upper = 0, 0
            
        ld_results.append({'LC/LD': f"LD{val}", 'Dose': dose_val, 'Lower Limit (95%)': lower, 'Upper Limit (95%)': upper})
    
    df_ld = pd.DataFrame(ld_results)

    results = {
        'slope': slope, 'intercept': intercept, 'ld50': ld50, 'ld50_log': ld50_log,
        'chi': chi_sq, 'df': df_deg, 'h': h_factor, 'r2': r_squared,
        'table_15': df_table, 'table_ld': df_ld, 'raw': final_vars
    }
    return results, None

# --- 3. HELPER: GRAPH GENERATORS ---
def plot_linear(res):
    fig, ax = plt.subplots(figsize=(8, 6))
    emp_disp = res['raw']['y_emp']
    x_log = res['raw']['x_log']
    
    ax.scatter(x_log, emp_disp, color='blue', s=60, label='Observed (Empirical)', zorder=3)
    x_line = np.linspace(min(x_log)-0.2, max(x_log)+0.2, 100)
    ax.plot(x_line, res['intercept'] + res['slope']*x_line, color='red', linewidth=2, label='Regression Line')
    
    # LD50 Markers
    ax.axhline(5, color='green', linestyle='--', alpha=0.6)
    ax.axvline(res['ld50_log'], color='green', linestyle='--', alpha=0.6)
    ax.text(res['ld50_log'], 5.1, f' LD50\n {res["ld50"]:.2f}', color='green', fontweight='bold')
    
    # Add R-Squared to plot
    ax.text(0.05, 0.95, f'$R^2 = {res["r2"]:.4f}$', transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title("Probit Analysis - Linear Regression Plot")
    ax.set_xlabel("Log Concentration (Dose)")
    ax.set_ylabel("Probit Value")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')
    plt.tight_layout()
    return fig

def plot_sigmoid(res):
    fig, ax = plt.subplots(figsize=(8, 6))
    p_obs = res['raw']['p_obs'] * 100
    x_log = res['raw']['x_log']
    
    ax.scatter(x_log, p_obs, color='black', s=60, label='Observed Mortality %', zorder=3)
    
    x_line = np.linspace(min(x_log)-0.2, max(x_log)+0.2, 100)
    y_smooth = res['intercept'] + res['slope'] * x_line
    p_smooth = norm.cdf(y_smooth - 5) * 100
    
    ax.plot(x_line, p_smooth, color='green', linewidth=2, label='Dose-Response Curve')
    
    # LD50 Markers
    ax.axhline(50, color='red', linestyle='--', alpha=0.6)
    ax.axvline(res['ld50_log'], color='red', linestyle='--', alpha=0.6)
    ax.text(res['ld50_log'], 55, f' LD50', color='red', fontweight='bold')
    
    ax.set_title("Standard Dose-Response Curve (Sigmoidal)")
    ax.set_xlabel("Log Concentration (Dose)")
    ax.set_ylabel("% Mortality")
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')
    plt.tight_layout()
    return fig

def convert_plot_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    return buf

# --- 4. PDF REPORT GENERATOR ---
def create_pdf(results, fig_linear, fig_sigmoid):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "VKC Probit Analysis Report")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, "Generated via Streamlit App")
    
    # Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 100, "1. Summary Statistics")
    c.setFont("Helvetica", 11)
    c.drawString(60, height - 120, f"Regression Equation:  Y = {results['intercept']:.4f} + {results['slope']:.4f} * Log(X)")
    c.drawString(60, height - 135, f"R-Squared (R²):       {results['r2']:.4f}")
    c.drawString(60, height - 150, f"LD50 Value:           {results['ld50']:.4f}")
    c.drawString(60, height - 165, f"Chi-Square:           {results['chi']:.4f} (df={results['df']})")
    
    # LD Table
    y_pos = height - 200
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_pos, "2. Lethal Concentrations (95% Limits)")
    y_pos -= 20
    
    c.setFont("Courier", 10)
    headers = f"{'Type':<8} {'Dose':<12} {'Lower':<12} {'Upper':<12}"
    c.drawString(50, y_pos, headers)
    c.line(50, y_pos-5, 350, y_pos-5)
    y_pos -= 15
    
    for _, row in results['table_ld'].iterrows():
        line = f"{row['LC/LD']:<8} {row['Dose']:<12.4f} {row['Lower Limit (95%)']:<12.4f} {row['Upper Limit (95%)']:<12.4f}"
        c.drawString(50, y_pos, line)
        y_pos -= 15

    # Images
    # Linear Plot
    y_pos -= 250
    if y_pos < 50: c.showPage(); y_pos = height - 300
    
    img_linear = ImageReader(convert_plot_to_image(fig_linear))
    c.drawImage(img_linear, 50, y_pos, width=400, height=300)
    c.drawString(50, y_pos + 305, "Fig 1: Linear Regression")

    # Sigmoid Plot
    y_pos -= 320
    if y_pos < 50: c.showPage(); y_pos = height - 300
    
    img_sigmoid = ImageReader(convert_plot_to_image(fig_sigmoid))
    c.drawImage(img_sigmoid, 50, y_pos, width=400, height=300)
    c.drawString(50, y_pos + 305, "Fig 2: Dose-Response Curve")

    c.save()
    buffer.seek(0)
    return buffer

# --- 5. MAIN UI LAYOUT ---
st.title("🧪 VKC Probit Analysis Tool")
st.markdown("Professional toxicology analysis using Finney's Iterative Method.")

# Sidebar
with st.sidebar:
    st.header("📂 Upload Data")
    uploaded_file = st.file_uploader("Upload Excel or CSV", type=['xlsx', 'csv'])
    st.caption("Ensure columns: **Dose**, **Total**, **Dead**")
    
    if uploaded_file:
        st.success("File Loaded!")

# Main Content
if uploaded_file:
    # Read File
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Run Analysis
    res, err = probit_analysis(df)
    
    if err:
        st.error(err)
    else:
        # Create Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Results & Tables", 
            "📈 Linear Graph", 
            "📉 Dose-Response Graph", 
            "📄 Full PDF Report"
        ])
        
        # --- TAB 1: RESULTS ---
        with tab1:
            st.subheader("Summary Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Slope (b)", f"{res['slope']:.4f}")
            c2.metric("Intercept (a)", f"{res['intercept']:.4f}")
            c3.metric("R-Squared", f"{res['r2']:.4f}")
            c4.metric("LD50 Value", f"{res['ld50']:.4f}", delta_color="normal")
            
            st.info(f"**Regression Equation:** Y = {res['intercept']:.4f} + {res['slope']:.4f} * Log(Dose)")
            
            st.write("### 1. Lethal Concentrations (with 95% Confidence Limits)")
            st.dataframe(res['table_ld'].style.format({"Dose": "{:.4f}", "Lower Limit (95%)": "{:.4f}", "Upper Limit (95%)": "{:.4f}"}), use_container_width=True)
            
            st.write("### 2. Full Calculation Table (15 Columns)")
            st.dataframe(res['table_15'].style.format("{:.4f}"), use_container_width=True)

        # --- TAB 2: LINEAR GRAPH ---
        with tab2:
            st.subheader("Linear Regression (Probit Scale)")
            fig_lin = plot_linear(res)
            st.pyplot(fig_lin)
            
            # Download Button for Linear Graph
            buf_lin = convert_plot_to_image(fig_lin)
            st.download_button(
                label="📥 Download Linear Graph (PNG)",
                data=buf_lin,
                file_name="linear_regression_plot.png",
                mime="image/png"
            )

        # --- TAB 3: SIGMOID GRAPH ---
        with tab3:
            st.subheader("Standard Dose-Response Curve")
            fig_sig = plot_sigmoid(res)
            st.pyplot(fig_sig)
            
            # Download Button for Sigmoid Graph
            buf_sig = convert_plot_to_image(fig_sig)
            st.download_button(
                label="📥 Download Dose-Response Graph (PNG)",
                data=buf_sig,
                file_name="dose_response_curve.png",
                mime="image/png"
            )

        # --- TAB 4: PDF REPORT ---
        with tab4:
            st.subheader("Generate Full Report")
            st.write("This PDF includes the Summary, R-Squared, Tables, and both Graphs.")
            
            # Generate PDF Button
            if st.button("Generate PDF Now"):
                with st.spinner("Generating PDF..."):
                    # We regenerate figs to ensure they are clean for PDF
                    f1 = plot_linear(res)
                    f2 = plot_sigmoid(res)
                    pdf_bytes = create_pdf(res, f1, f2)
                    
                    st.success("PDF Generated!")
                    st.download_button(
                        label="📥 Click to Download PDF Report",
                        data=pdf_bytes,
                        file_name="VKC_Probit_Report.pdf",
                        mime="application/pdf"
                    )

else:
    st.info("👋 Please upload a dataset in the Sidebar to begin.")