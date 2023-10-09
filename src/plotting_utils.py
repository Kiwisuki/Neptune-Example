from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier, XGBRegressor, plot_importance


def scatter_residual_analysis(data: pd.DataFrame) -> go.Figure:
    """Creates an animated scatter plot to visualize errors in predictions.

    Args:
        data: pandas dataframe with columns 'alcohol', 'volatile acidity', 'quality', 'predicted_quality'

    Returns:
        fig: plotly figure
    """
    data = data.sort_values(by='quality')
    data['residual_size'] = data['predicted_quality'] - data['quality']

    x_min, x_max = data['alcohol'].min(), data['alcohol'].max()
    y_min, y_max = data['volatile acidity'].min(), data['volatile acidity'].max()

    fig = px.scatter(
        data,
        x='alcohol',
        y='volatile acidity',
        animation_frame='quality',
        color='residual_size',
        size='sulphates',
        color_continuous_scale='RdBu',
        range_color=[-2, 2],
        hover_data=['residual_size'],
    )
    fig.update_layout(plot_bgcolor='rgb(122, 122, 122)')
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])
    fig.update_layout(title='Residual analysis - Highest corellation features')
    return fig


def plot_feature_importance(model: Union[XGBRegressor, XGBClassifier]) -> plt.Figure:
    """Plots feature importance for a given (xgb)model."""
    fig, ax = plt.subplots(figsize=(15, 15))
    plot_importance(model, ax=ax)
    fig.set_dpi(400)
    return fig
