"""
Visualization utilities for Endo Digital Twin.
This module contains functions for creating charts and visualizations.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Any

class EndoVisualizer:
    """
    Visualization utilities for endometriosis data.
    """
    
    def __init__(self):
        self.color_primary = '#6B46C1'  # Purple
        self.color_secondary = '#14B8A6'  # Teal
        self.color_background = '#F8FAFC'
    
    def create_pain_timeline(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a timeline chart showing pain levels over time.
        
        Args:
            df: DataFrame with 'date' and 'pain_level' columns
        
        Returns:
            go.Figure: Plotly figure
        """
        fig = px.line(
            df, 
            x='date', 
            y='pain_level',
            title='Pain Level Over Time',
            labels={'pain_level': 'Pain Level (0-10)', 'date': 'Date'},
            color_discrete_sequence=[self.color_primary]
        )
        
        fig.update_traces(line_width=3, marker_size=6)
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font=dict(color='#2D3748', size=16),
            font=dict(color='#2D3748', size=12),
            xaxis=dict(title_font=dict(color='#2D3748', size=12)),
            yaxis=dict(title_font=dict(color='#2D3748', size=12))
        )
        
        return fig
    
    def create_pain_distribution(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a histogram showing pain level distribution.
        
        Args:
            df: DataFrame with 'pain_level' column
        
        Returns:
            go.Figure: Plotly figure
        """
        fig = px.histogram(
            df, 
            x='pain_level', 
            nbins=11,
            title='Pain Level Distribution',
            labels={'pain_level': 'Pain Level (0-10)', 'count': 'Frequency'},
            color_discrete_sequence=[self.color_secondary]
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font=dict(color='#2D3748', size=16),
            font=dict(color='#2D3748', size=12),
            xaxis=dict(title_font=dict(color='#2D3748', size=12)),
            yaxis=dict(title_font=dict(color='#2D3748', size=12))
        )
        
        return fig
    
    def create_comparison_chart(self, baseline: float, simulated: float) -> go.Figure:
        """
        Create a bar chart comparing baseline vs simulated pain levels.
        
        Args:
            baseline: Baseline pain level
            simulated: Simulated pain level
        
        Returns:
            go.Figure: Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Baseline',
            x=['Pain Level'],
            y=[baseline],
            marker_color=self.color_primary,
            text=[f'{baseline:.1f}'],
            textposition='auto',
            textfont=dict(color='white', size=14)
        ))
        
        fig.add_trace(go.Bar(
            name='Simulated',
            x=['Pain Level'],
            y=[simulated],
            marker_color=self.color_secondary,
            text=[f'{simulated:.1f}'],
            textposition='auto',
            textfont=dict(color='white', size=14)
        ))
        
        fig.update_layout(
            title="Pain Level Comparison",
            xaxis_title="Scenario",
            yaxis_title="Pain Level (0-10)",
            yaxis=dict(range=[0, 10], title_font=dict(color='#2D3748', size=12)),
            height=400,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font=dict(color='#2D3748', size=16),
            font=dict(color='#2D3748', size=12),
            xaxis=dict(title_font=dict(color='#2D3748', size=12)),
            legend=dict(font=dict(color='#2D3748', size=12))
        )
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a correlation heatmap for numeric features.
        
        Args:
            df: DataFrame with numeric columns
        
        Returns:
            go.Figure: Plotly figure
        """
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        
        fig.update_layout(
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font=dict(color='#2D3748', size=16),
            font=dict(color='#2D3748', size=12)
        )
        
        return fig
    
    def create_mood_timeline(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a timeline chart showing mood over time.
        
        Args:
            df: DataFrame with 'date' and 'mood' columns
        
        Returns:
            go.Figure: Plotly figure
        """
        # Convert mood to numeric for plotting
        mood_mapping = {
            'Terrible': 1,
            'Poor': 2,
            'Okay': 3,
            'Good': 4,
            'Excellent': 5
        }
        
        df_plot = df.copy()
        df_plot['mood_numeric'] = df_plot['mood'].map(mood_mapping)
        
        fig = px.line(
            df_plot,
            x='date',
            y='mood_numeric',
            title='Mood Over Time',
            labels={'mood_numeric': 'Mood Level', 'date': 'Date'},
            color_discrete_sequence=[self.color_primary]
        )
        
        # Update y-axis to show mood labels
        fig.update_yaxes(
            tickmode='array',
            tickvals=list(mood_mapping.values()),
            ticktext=list(mood_mapping.keys())
        )
        
        fig.update_traces(line_width=3, marker_size=6)
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font=dict(color='#2D3748', size=16),
            font=dict(color='#2D3748', size=12),
            xaxis=dict(title_font=dict(color='#2D3748', size=12)),
            yaxis=dict(title_font=dict(color='#2D3748', size=12))
        )
        
        return fig

