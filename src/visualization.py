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
    
    def create_feature_impact_chart(self, feature_impacts: Dict[str, float]) -> go.Figure:
        """
        Create horizontal bar chart showing feature impacts on pain.
        
        Args:
            feature_impacts: Dictionary with feature names and their impact values
        
        Returns:
            go.Figure: Plotly figure
        """
        # Sort by absolute impact (descending)
        sorted_features = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
        
        features = [f[0].replace('_', ' ').title() for f in sorted_features]
        impacts = [f[1] for f in sorted_features]
        
        colors = [self.color_secondary if impact < 0 else '#ef4444' for impact in impacts]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=features,
            x=impacts,
            orientation='h',
            marker_color=colors,
            text=[f"{impact:+.2f}" for impact in impacts],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="What Influenced Your Pain Prediction?",
            xaxis_title="Pain Change (points)",
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font=dict(color='#2D3748', size=14),
            font=dict(color='#2D3748', size=11),
            xaxis=dict(title_font=dict(color='#2D3748', size=11)),
            yaxis=dict(
                title_font=dict(color='#2D3748', size=11),
                tickfont=dict(color='#1a202c', size=11)
            )
        )
        
        return fig
    
    def create_pain_profile_radar(self, user_values: Dict[str, float], 
                                   ideal_values: Dict[str, float]) -> go.Figure:
        """
        Create radar chart comparing user profile vs ideal.
        
        Args:
            user_values: Dictionary with user's current values
            ideal_values: Dictionary with ideal/target values
        
        Returns:
            go.Figure: Plotly figure
        """
        categories = list(user_values.keys())
        user_data = [user_values[cat] for cat in categories]
        ideal_data = [ideal_values[cat] for cat in categories]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=user_data + [user_data[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name='You',
            line_color=self.color_secondary,
            fillcolor=self.color_secondary,
            opacity=0.6
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=ideal_data + [ideal_data[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Ideal Balance',
            line_color='#9ca3af',
            fillcolor='#9ca3af',
            opacity=0.3
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickfont=dict(color='white', size=11),
                    linecolor='#9ca3af'
                ),
                angularaxis=dict(
                    tickfont=dict(color='white', size=11),
                    linecolor='#9ca3af'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            title="Your Daily Profile vs Ideal Balance",
            height=350,
            title_font=dict(color='#2D3748', size=14),
            font=dict(color='#2D3748', size=11),
            legend=dict(font=dict(color='#2D3748', size=11))
        )
        
        return fig
    
    def create_wellbeing_gauge(self, score: float) -> go.Figure:
        """
        Create gauge chart showing overall lifestyle balance score.
        
        Args:
            score: Score from 0-100
        
        Returns:
            go.Figure: Plotly figure
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Well-being Balance Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': self.color_secondary},
                'steps': [
                    {'range': [0, 50], 'color': '#fee2e2'},
                    {'range': [50, 75], 'color': '#fef3c7'},
                    {'range': [75, 100], 'color': '#dcfce7'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=350,
            font=dict(color='#2D3748', size=11),
            title_font=dict(color='#2D3748', size=14)
        )
        
        return fig

