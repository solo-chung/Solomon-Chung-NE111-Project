import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from io import StringIO
import seaborn as sns

# Set page config
st.set_page_config(page_title="Distribution Fitting Tool", layout="wide", page_icon="📊")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 8px 16px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.3);
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #667eea;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📊 Distribution Fitting Tool")
st.markdown("**NE111 Project - Statistical Distribution Analysis**")
st.markdown("---")

# Available distributions
DISTRIBUTIONS = {
    'Normal': stats.norm,
    'Gamma': stats.gamma,
    'Weibull': stats.weibull_min,
    'Exponential': stats.expon,
    'Log-Normal': stats.lognorm,
    'Beta': stats.beta,
    'Chi-Square': stats.chi2,
    'Uniform': stats.uniform,
    'Rayleigh': stats.rayleigh,
    'Pareto': stats.pareto,
    'Cauchy': stats.cauchy,
    'Laplace': stats.laplace
}

# Color palette
COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#fa709a', '#fee140']

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'fitted_params' not in st.session_state:
    st.session_state.fitted_params = {}

# Sidebar for data input
with st.sidebar:
    st.header("📥 Data Input")
    
    input_method = st.radio("Choose input method:", ["Manual Entry", "Upload CSV", "Generate Sample"])
    
    if input_method == "Manual Entry":
        st.subheader("Enter comma-separated values:")
        data_input = st.text_area("Data:", height=150, placeholder="1.2, 3.4, 5.6, 7.8, ...")
        
        if st.button("Load Data", type="primary"):
            try:
                data_list = [float(x.strip()) for x in data_input.split(',') if x.strip()]
                if len(data_list) > 0:
                    st.session_state.data = np.array(data_list)
                    st.success(f"✅ Loaded {len(data_list)} data points!")
                else:
                    st.error("Please enter valid numbers")
            except ValueError:
                st.error("Invalid input! Use comma-separated numbers.")
    
    elif input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        col_name = st.text_input("Column name (leave empty for first column):", "")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview:", df.head())
                
                if st.button("Load Data", type="primary"):
                    if col_name and col_name in df.columns:
                        st.session_state.data = df[col_name].dropna().values
                    else:
                        st.session_state.data = df.iloc[:, 0].dropna().values
                    st.success(f"✅ Loaded {len(st.session_state.data)} data points!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    else:  # Generate Sample
        st.subheader("Generate sample data:")
        sample_dist = st.selectbox("Distribution:", list(DISTRIBUTIONS.keys()))
        sample_size = st.slider("Sample size:", 100, 5000, 1000)
        
        if st.button("Generate Data", type="primary"):
            if sample_dist == 'Normal':
                st.session_state.data = np.random.normal(10, 2, sample_size)
            elif sample_dist == 'Gamma':
                st.session_state.data = stats.gamma.rvs(5, 1, 1, size=sample_size)
            elif sample_dist == 'Weibull':
                st.session_state.data = stats.weibull_min.rvs(1.5, size=sample_size)
            elif sample_dist == 'Exponential':
                st.session_state.data = np.random.exponential(2, sample_size)
            else:
                st.session_state.data = np.random.normal(10, 2, sample_size)
            st.success(f"✅ Generated {sample_size} data points!")

# Main content
if st.session_state.data is not None:
    data = st.session_state.data
    
    # Display data statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Sample Size", len(data))
    with col2:
        st.metric("📈 Mean", f"{np.mean(data):.3f}")
    with col3:
        st.metric("📉 Std Dev", f"{np.std(data):.3f}")
    with col4:
        st.metric("🎯 Range", f"{np.ptp(data):.3f}")
    
    st.markdown("---")
    
    # Tabs for different features
    tab1, tab2, tab3 = st.tabs(["🎯 Auto Fit", "🎚️ Manual Fit", "📋 Data View"])
    
    with tab1:
        st.subheader("Automatic Distribution Fitting")
        
        col_left, col_right = st.columns([2, 1])
        
        with col_right:
            st.markdown("### Settings")
            selected_dist = st.selectbox("Select Distribution:", list(DISTRIBUTIONS.keys()))
            bins = st.slider("Histogram Bins:", 10, 100, 30)
            show_kde = st.checkbox("Show KDE", value=True)
            
            if st.button("🔍 Fit Distribution", type="primary", use_container_width=True):
                try:
                    dist = DISTRIBUTIONS[selected_dist]
                    params = dist.fit(data)
                    st.session_state.fitted_params[selected_dist] = params
                    st.success(f"✅ Fitted {selected_dist} distribution!")
                except Exception as e:
                    st.error(f"Fitting error: {e}")
            
            # Display fitted parameters
            if selected_dist in st.session_state.fitted_params:
                st.markdown("### 📊 Fitted Parameters")
                params = st.session_state.fitted_params[selected_dist]
                
                param_names = ['Shape', 'Location', 'Scale']
                for i, param in enumerate(params):
                    if i < len(param_names):
                        st.metric(param_names[i], f"{param:.4f}")
                    else:
                        st.metric(f"Param {i+1}", f"{param:.4f}")
                
                # Calculate fit quality
                dist_obj = DISTRIBUTIONS[selected_dist](*params)
                hist, bin_edges = np.histogram(data, bins=bins, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                expected = dist_obj.pdf(bin_centers)
                
                mse = np.mean((hist - expected) ** 2)
                max_error = np.max(np.abs(hist - expected))
                
                st.markdown("### 📈 Fit Quality")
                st.metric("Mean Squared Error", f"{mse:.6f}")
                st.metric("Max Error", f"{max_error:.4f}")
        
        with col_left:
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
            
            # Histogram
            n, bins_plot, patches = ax.hist(data, bins=bins, density=True, alpha=0.6, 
                                           color=COLORS[0], edgecolor='white', linewidth=1.5)
            
            # Color gradient for bars
            for i, patch in enumerate(patches):
                patch.set_facecolor(plt.cm.viridis(i / len(patches)))
            
            # KDE
            if show_kde:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x_kde = np.linspace(data.min(), data.max(), 200)
                ax.plot(x_kde, kde(x_kde), linewidth=2.5, color=COLORS[1], 
                       label='KDE', linestyle='--')
            
            # Fitted distribution
            if selected_dist in st.session_state.fitted_params:
                params = st.session_state.fitted_params[selected_dist]
                dist_obj = DISTRIBUTIONS[selected_dist](*params)
                x_range = np.linspace(data.min(), data.max(), 200)
                y_fit = dist_obj.pdf(x_range)
                ax.plot(x_range, y_fit, linewidth=3, color=COLORS[4], 
                       label=f'{selected_dist} Fit')
            
            ax.set_xlabel('Value', fontsize=12, fontweight='bold')
            ax.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax.set_title(f'Distribution Fitting: {selected_dist}', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(frameon=True, shadow=True, fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)
    
    with tab2:
        st.subheader("Manual Parameter Adjustment")
        
        manual_dist = st.selectbox("Distribution:", list(DISTRIBUTIONS.keys()), key='manual')
        
        col_plot, col_sliders = st.columns([2, 1])
        
        with col_sliders:
            st.markdown("### Adjust Parameters")
            
            # Get initial parameters
            if manual_dist in st.session_state.fitted_params:
                init_params = st.session_state.fitted_params[manual_dist]
            else:
                try:
                    dist = DISTRIBUTIONS[manual_dist]
                    init_params = dist.fit(data)
                except:
                    init_params = [1.0, 0.0, 1.0]
            
            # Create sliders based on distribution
            manual_params = []
            param_labels = ['Shape', 'Location', 'Scale']
            
            for i in range(len(init_params)):
                label = param_labels[i] if i < len(param_labels) else f"Param {i+1}"
                init_val = float(init_params[i])
                
                # Set reasonable ranges
                if 'Location' in label:
                    min_val, max_val = data.min() - 5, data.max() + 5
                elif 'Scale' in label:
                    min_val, max_val = 0.1, 10.0
                else:  # Shape
                    min_val, max_val = 0.1, 10.0
                
                val = st.slider(label, min_val, max_val, init_val, 0.1, key=f'slider_{i}')
                manual_params.append(val)
            
            bins_manual = st.slider("Histogram Bins:", 10, 100, 30, key='bins_manual')
        
        with col_plot:
            # Plot with manual parameters
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
            
            n, bins_plot, patches = ax.hist(data, bins=bins_manual, density=True, 
                                           alpha=0.6, color=COLORS[2], edgecolor='white', 
                                           linewidth=1.5)
            
            # Color gradient
            for i, patch in enumerate(patches):
                patch.set_facecolor(plt.cm.plasma(i / len(patches)))
            
            try:
                dist_obj = DISTRIBUTIONS[manual_dist](*manual_params)
                x_range = np.linspace(data.min(), data.max(), 200)
                y_fit = dist_obj.pdf(x_range)
                ax.plot(x_range, y_fit, linewidth=3, color=COLORS[5], 
                       label=f'{manual_dist} (Manual)')
            except Exception as e:
                st.error(f"Invalid parameters: {e}")
            
            ax.set_xlabel('Value', fontsize=12, fontweight='bold')
            ax.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax.set_title(f'Manual Fitting: {manual_dist}', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(frameon=True, shadow=True, fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Statistical Summary")
            summary_df = pd.DataFrame({
                'Metric': ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max'],
                'Value': [
                    len(data),
                    np.mean(data),
                    np.std(data),
                    np.min(data),
                    np.percentile(data, 25),
                    np.percentile(data, 50),
                    np.percentile(data, 75),
                    np.max(data)
                ]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 📈 Data Preview")
            preview_df = pd.DataFrame({'Values': data[:50]})
            st.dataframe(preview_df, use_container_width=True, height=300)
        
        # Download option
        st.markdown("### 💾 Export Data")
        csv = pd.DataFrame({'data': data}).to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="fitted_data.csv",
            mime="text/csv",
            type="primary"
        )

else:
    # Welcome screen
    st.info("👈 Please load data using the sidebar to begin!")
    
    st.markdown("""
    ### Welcome to the Distribution Fitting Tool! 🎉
    
    This app allows you to:
    - 📊 Load data manually, from CSV, or generate samples
    - 🎯 Automatically fit 12+ statistical distributions
    - 🎚️ Manually adjust parameters with interactive sliders
    - 📈 Visualize data with beautiful, colorful plots
    - 📉 Evaluate fit quality with error metrics
    
    **Get started by selecting a data input method in the sidebar!**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white;'>
    <p><b>NE111 - Introduction to Python | University of Waterloo</b></p>
    <p>Built with Streamlit 🎈 | Powered by SciPy & Matplotlib</p>
</div>
""", unsafe_allow_html=True)