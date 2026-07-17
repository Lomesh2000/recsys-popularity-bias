import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
import pandas as pd
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="ILE-BPR Fairness Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .recommendation-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .niche-item {
        background: #fff3cd;
        border-left-color: #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎯 ILE-BPR: Fairness-Aware Recommender System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Item Loss Equalization for Mitigating Popularity Bias</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📊 Configuration")
lambda_values = [0.0, 0.05, 0.10, 0.25, 0.50, 1.0]
selected_lambda = st.sidebar.selectbox("Select λ (Fairness Penalty)", lambda_values, index=3)

# Load data
@st.cache_data
def load_pareto_data():
    try:
        with open('pareto_results.json', 'r') as f:
            return json.load(f)
    except:
        return []

@st.cache_data
def load_training_history(lambda_val):
    run_name = 'bpr_baseline' if lambda_val == 0.0 else f'ile_std_lambda_{lambda_val}'
    history_path = f'./check/{run_name}/history.json'
    try:
        with open(history_path, 'r') as f:
            return json.load(f)
    except:
        return None

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📈 Pareto Frontier", "🔄 Training Dynamics", "🎬 Niche User Analysis", "📊 Model Comparison"])

# Tab 1: Pareto Frontier
with tab1:
    st.header("Accuracy-Fairness Trade-off")
    
    pareto_data = load_pareto_data()
    if pareto_data:
        df = pd.DataFrame(pareto_data)
        
        # Create interactive Pareto plot
        fig = go.Figure()
        
        # Add trajectory line
        fig.add_trace(go.Scatter(
            x=df['nDCG'], y=df['UPD'],
            mode='lines',
            name='Trade-off Curve',
            line=dict(color='gray', dash='dash', width=2)
        ))
        
        # Add points
        colors = ['red' if lam == 0.0 else 'green' if lam == 0.25 else 'blue' for lam in df['lambda']]
        fig.add_trace(go.Scatter(
            x=df['nDCG'], y=df['UPD'],
            mode='markers+text',
            marker=dict(size=15, color=colors, line=dict(width=2, color='black')),
            text=[f'λ={lam}' for lam in df['lambda']],
            textposition="top center",
            name='Lambda Values',
            hovertemplate='<b>λ=%{text}</b><br>nDCG: %{x:.4f}<br>UPD: %{y:.4f}<extra></extra>'
        ))
        
        # Highlight optimal point
        optimal = df[df['lambda'] == 0.25]
        if not optimal.empty:
            fig.add_trace(go.Scatter(
                x=optimal['nDCG'], y=optimal['UPD'],
                mode='markers',
                marker=dict(size=25, color='rgba(255,0,0,0)', line=dict(width=3, color='red')),
                name='Pareto Optimal (λ=0.25)'
            ))
        
        fig.update_layout(
            title='nDCG vs UPD: Finding the Sweet Spot',
            xaxis_title='Accuracy (nDCG@10) → Higher is Better',
            yaxis_title='Unfairness (UPD) → Lower is Better',
            hovermode='closest',
            showlegend=True,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics cards
        col1, col2, col3, col4 = st.columns(4)
        baseline = df[df['lambda'] == 0.0].iloc[0] if len(df[df['lambda'] == 0.0]) > 0 else None
        optimal = df[df['lambda'] == 0.25].iloc[0] if len(df[df['lambda'] == 0.25]) > 0 else None
        
        if baseline is not None and optimal is not None:
            with col1:
                ndcg_drop = ((baseline['nDCG'] - optimal['nDCG']) / baseline['nDCG']) * 100
                st.metric("nDCG Drop", f"{ndcg_drop:.1f}%", delta=f"-{ndcg_drop:.1f}%")
            with col2:
                upd_improvement = ((baseline['UPD'] - optimal['UPD']) / baseline['UPD']) * 100
                st.metric("UPD Improvement", f"{upd_improvement:.1f}%", delta=f"+{upd_improvement:.1f}%")
            with col3:
                ad_improvement = ((optimal['AD'] - baseline['AD']) / baseline['AD']) * 100
                st.metric("AD Improvement", f"{ad_improvement:.1f}%", delta=f"+{ad_improvement:.1f}%")
            with col4:
                ee_improvement = ((optimal['EE'] - baseline['EE']) / baseline['EE']) * 100
                st.metric("EE Improvement", f"{ee_improvement:.1f}%", delta=f"+{ee_improvement:.1f}%")

# Tab 2: Training Dynamics
with tab2:
    st.header(f"Training Dynamics (λ = {selected_lambda})")
    
    history = load_training_history(selected_lambda)
    if history:
        # Loss plot
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=history['train_loss_per_epoch'],
            mode='lines',
            name='Training Loss',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_loss.update_layout(
            title='Training Loss Over Epochs',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            height=400
        )
        st.plotly_chart(fig_loss, use_container_width=True)
        
        # Metrics plot
        if history.get('val_metrics'):
            val_df = pd.DataFrame(history['val_metrics'])
            
            fig_metrics = go.Figure()
            metrics_to_plot = ['nDCG', 'UPD', 'AD', 'EE']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for metric, color in zip(metrics_to_plot, colors):
                if metric in val_df.columns:
                    fig_metrics.add_trace(go.Scatter(
                        x=val_df['epoch'],
                        y=val_df[metric],
                        mode='lines+markers',
                        name=metric,
                        line=dict(color=color, width=2),
                        marker=dict(size=6)
                    ))
            
            fig_metrics.update_layout(
                title='Validation Metrics Over Epochs',
                xaxis_title='Epoch',
                yaxis_title='Metric Value',
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
    else:
        st.warning(f"No training history found for λ = {selected_lambda}")

# Tab 3: Niche User Analysis
with tab3:
    st.header("🎬 Niche User Recommendation Analysis")
    st.markdown("Compare how ILE changes recommendations for users with non-mainstream tastes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("❌ Standard BPR (λ=0.0)")
        st.markdown("**User 5943 Profile:** 41.7% Head, 41.7% Mid, 16.7% Tail")
        st.markdown("#### Recommendations:")
        
        bpr_recs = [
            {"rank": 1, "item": "Movie 3203", "group": "H", "score": 3.09},
            {"rank": 2, "item": "Movie 837", "group": "H", "score": 3.02},
            {"rank": 3, "item": "Movie 869", "group": "H", "score": 2.93},
            {"rank": 4, "item": "Movie 884", "group": "M", "score": 2.92},
            {"rank": 5, "item": "Movie 855", "group": "H", "score": 2.68},
        ]
        
        for rec in bpr_recs:
            bg_class = "recommendation-card" if rec['group'] == 'H' else "recommendation-card niche-item"
            st.markdown(f"""
            <div class="{bg_class}">
                <b>#{rec['rank']}</b> {rec['item']} <span style="float:right">{rec['group']} | Score: {rec['score']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.metric("Head Items", "65%", delta="Over-represented")
    
    with col2:
        st.subheader("✅ ILE-BPR (λ=0.25)")
        st.markdown("**User 5943 Profile:** 41.7% Head, 41.7% Mid, 16.7% Tail")
        st.markdown("#### Recommendations:")
        
        ile_recs = [
            {"rank": 1, "item": "Movie 2442", "group": "M", "score": 4.25},
            {"rank": 2, "item": "Movie 2452", "group": "M", "score": 4.09},
            {"rank": 3, "item": "Movie 2446", "group": "M", "score": 3.89},
            {"rank": 4, "item": "Movie 1212", "group": "H", "score": 3.86},
            {"rank": 5, "item": "Movie 1186", "group": "H", "score": 3.83},
        ]
        
        for rec in ile_recs:
            bg_class = "recommendation-card" if rec['group'] == 'H' else "recommendation-card niche-item"
            st.markdown(f"""
            <div class="{bg_class}">
                <b>#{rec['rank']}</b> {rec['item']} <span style="float:right">{rec['group']} | Score: {rec['score']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.metric("Mid Items", "60%", delta="+25% improvement")
    
    st.markdown("---")
    st.info("""
    **Key Insight:** ILE successfully elevates Mid-tier items for niche users, 
    bringing recommendations closer to their actual preferences (41.7% Mid) 
    instead of defaulting to popular Head items.
    """)

# Tab 4: Model Comparison
with tab4:
    st.header("📊 Comprehensive Model Comparison")
    
    if pareto_data:
        df = pd.DataFrame(pareto_data)
        
        # Comparison table
        st.subheader("All Lambda Values")
        display_df = df.copy()
        display_df.columns = ['λ', 'nDCG@10', 'UPD', 'AD', 'EE']
        st.dataframe(
            display_df.style.format({
                'nDCG@10': '{:.4f}',
                'UPD': '{:.4f}',
                'AD': '{:.4f}',
                'EE': '{:.4f}'
            }).highlight_max(subset=['nDCG@10', 'AD', 'EE'], color='#d4edda')
             .highlight_min(subset=['UPD'], color='#d4edda'),
            use_container_width=True
        )
        
        # Radar chart for optimal lambda
        st.subheader("Performance Radar (λ=0.25)")
        optimal = df[df['lambda'] == 0.25].iloc[0]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[optimal['nDCG'], 1-optimal['UPD'], optimal['AD'], optimal['EE']],
            theta=['nDCG', 'Fairness (1-UPD)', 'Diversity (AD)', 'Equality (EE)'],
            fill='toself',
            name='ILE-BPR (λ=0.25)'
        ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=500
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with Streamlit | ILE-BPR Fairness Recommender System</p>
    <p><a href="https://github.com/yourusername/ile-bpr" target="_blank">View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)